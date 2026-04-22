import torch
from contextlib import nullcontext
from typing import Union, List, Callable, Any, Dict, Generator, Optional, Type

from .utils import process_batch_data
from .event import Event, EventBroker
from .plugins.recorder import RecorderHub

from .utils.lifecycle import exit_manager

from dataclasses import dataclass

@dataclass
class Space:
    model: List[torch.nn.Module]
    optimizer: List[torch.optim.Optimizer]
    criterion: List[Callable]

class Engine:
    def __init__(
        self,
        accumulate_grad_batches: int = 1,
        precision: str = '32',
        clip_grad: bool = False,
        clip_max_norm: float = 1.0
    ):
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.clip_grad = clip_grad
        self.clip_max_norm = clip_max_norm

        self._moc: Dict[str, Dict[
            str, Union[
                List[torch.nn.Module],
                List[torch.optim.Optimizer],
                List[Callable]
            ]
        ]] = {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler(enabled=(self.precision == '16' and self.device.type == 'cuda'))
        self.broker = EventBroker()

        self.data: Any = None
        self.target: Any = None

        self.step_micro: int = 0
        self.step_micro_in_batch: int = 0
        self.step_global: int = 0
        self.step_global_in_batch: int = 0

        self.epoch: int = 0
        self._init_specs: Dict[str, List[Dict[str, Any]]] = {
            'optimizer': [],
            'criterion': [],
        }
        self._plugins: List[Any] = []
        self._recorder: RecorderHub = RecorderHub(self)

        self.mode = 'train'

        exit_manager.register(self._close)

    @property
    def is_training(self) -> bool:
        return self.mode == 'train'
    
    @property
    def plugins(self) -> List[Any]:
        return list(self._plugins)
    
    @property
    def recorder(self) -> RecorderHub:
        return self._recorder
    
    def set_recorder(
        self,
        path: str = '',
        name: str = '',
        resume: bool = False,
        auto_restore: bool = True,
        **kwargs
    ) -> 'Engine':
        '''
        Initialize and configure the recorder.

        Args:
            path (str): Custom root path for recordings.
            name (str): Experiment name.
            resume (bool): Whether to resume from existing recording.
            auto_restore (bool): Whether to auto-restore last checkpoint if available.
            **kwargs: Additional configuration options.

        Returns:
            Engine: Self for method chaining.
        '''
        self._recorder.set_recorder(path=path, name=name, resume=resume, **kwargs)
        
        if auto_restore:
            latest_ckpt = self._recorder.get_latest_checkpoint()
            if latest_ckpt is not None:
                state = self._recorder.load_checkpoint(latest_ckpt.name)
                self._restore_from_checkpoint(state)
        
        return self
    
    def _restore_from_checkpoint(self, state: Dict[str, Any]) -> None:
        '''
        Restore engine state from a checkpoint dictionary.

        Args:
            state (Dict[str, Any]): The checkpoint state dictionary.
        '''
        if 'epoch' in state:
            self.epoch = state['epoch']
        if 'step_global' in state:
            self.step_global = state['step_global']
        if 'step_micro' in state:
            self.step_micro = state['step_micro']
        
        for space_name, space_data in state.get('spaces', {}).items():
            if space_name not in self._moc:
                continue
            
            for i, model_state in enumerate(space_data.get('models', [])):
                if i < len(self._moc[space_name]['model']):
                    self._moc[space_name]['model'][i].load_state_dict(model_state)
            
            for i, opt_state in enumerate(space_data.get('optimizers', [])):
                if i < len(self._moc[space_name]['optimizer']):
                    self._moc[space_name]['optimizer'][i].load_state_dict(opt_state)
    
    @property
    def init_specs(self) -> Dict[str, Any]:
        return {
            'engine': {
                'accumulate_grad_batches': self.accumulate_grad_batches,
                'precision': self.precision,
                'clip_grad': self.clip_grad,
                'clip_max_norm': self.clip_max_norm,
                'device': str(self.device),
            },
            'spaces': {
                sp: {
                    'models': [m.__class__.__name__ for m in d['model']],
                    'n_optimizers': len(d['optimizer']),
                    'n_criterions': len(d['criterion']),
                }
                for sp, d in self._moc.items()
            },
            'optimizer': list(self._init_specs['optimizer']),
            'criterion': list(self._init_specs['criterion']),
        }

    def set_models_mode(self, mode: str):
        if mode not in ['train', 'eval']: return
        self.mode = mode
        is_train = (mode == 'train')
        for space in self._moc.values():
            for model in space['model']:
                model.train() if is_train else model.eval()

    def space(self, name: str) -> Space:
        if name not in self._moc:
            self._moc[name] = {'model': [], 'optimizer': [], 'criterion': []}
        return Space(
            model=self._moc[name]['model'],
            optimizer=self._moc[name]['optimizer'],
            criterion=self._moc[name]['criterion']
        )
    
    def find_plugin(self, target: Union[str, Type]) -> List[Any]:
        if isinstance(target, str):
            return [p for p in self._plugins if p.__class__.__name__ == target]
        return [p for p in self._plugins if isinstance(p, target)]
    
    def _collect_handlers(self, subscriber: Any):
        once_events = set(getattr(subscriber, '__once_events__', ()))
        persistent, once_map = {}, {}
        for attr in dir(subscriber):
            if not attr.startswith('on_'):
                continue
            fn = getattr(subscriber, attr, None)
            if not callable(fn):
                continue
            event_name = attr[3:]
            is_once = getattr(fn, '__once__', False) or event_name in once_events
            (once_map if is_once else persistent)[event_name] = fn
        return persistent, once_map
    
    def attach(self, subscriber: Union[Any, List[Any]]) -> 'Engine':
        subs = subscriber if isinstance(subscriber, list) else [subscriber]
        for s in subs:
            allow_dup = getattr(s, '__allow_duplicate__', False)
            if not allow_dup:
                if any(type(p) is type(s) for p in self._plugins):
                    continue
            if s in self._plugins:
                continue
            persistent, once_map = self._collect_handlers(s)
            for name, fn in persistent.items():
                self.broker.subscribe(name, fn)
            for name, fn in once_map.items():
                self.broker.subscribe_once(name, fn)
            self._plugins.append(s)
            if hasattr(s, 'setup') and callable(s.setup):
                s.setup(self)
        return self
    
    def detach(self, subscriber: Union[Any, str, List[Union[Any, str]]]) -> 'Engine':
        subs = subscriber if isinstance(subscriber, list) else [subscriber]
        targets: List[Any] = []
        for s in subs:
            if isinstance(s, str):
                targets.extend(self.find_plugin(s))
            else:
                targets.append(s)
        for s in targets:
            if s not in self._plugins:
                continue
            persistent, once_map = self._collect_handlers(s)
            for name, fn in persistent.items():
                self.broker.unsubscribe(name, fn)
            for name, fn in once_map.items():
                self.broker.unsubscribe(name, fn)
            if hasattr(s, 'teardown') and callable(s.teardown):
                s.teardown(self)
            self._plugins.remove(s)
        return self

    def emit(self, event_name: str, sender: str = 'engine', data: Any = None) -> 'Engine':
        self.broker.emit(Event(
            info=event_name,
            sender=sender,
            data=data,
            engine=self
        ))
        return self

    def add_model(self, model: Union[torch.nn.Module, List[torch.nn.Module]], space: str = 'default') -> 'Engine':
        if space not in self._moc:
            self._moc[space] = {'model': [], 'optimizer': [], 'criterion': []}
        if not isinstance(model, list):
            model = [model]
        for m in model:
            if not isinstance(m, torch.nn.Module): continue
            self._moc[space]['model'].append(m)
        return self

    def add_optimizer(self, optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]], space: str = 'default') -> 'Engine':
        if space not in self._moc:
            self._moc[space] = {'model': [], 'optimizer': [], 'criterion': []}
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for o in optimizer:
            if isinstance(o, torch.optim.Optimizer):
                self._moc[space]['optimizer'].append(o)
        return self

    def add_criterion(self, criterion: Union[Callable, List[Callable]], space: str = 'default') -> 'Engine':
        if space not in self._moc:
            self._moc[space] = {'model': [], 'optimizer': [], 'criterion': []}
        if not isinstance(criterion, list):
            criterion = [criterion]
        for c in criterion:
            self._moc[space]['criterion'].append(c)
        return self

    def new_optimizer(
        self,
        space: str,
        opt_type: Union[str, Type[torch.optim.Optimizer]],
        **kwargs,
    ) -> 'Engine':
        models = self._moc.get(space, {}).get('model', [])
        if not models:
            raise ValueError(f"space '{space}' 下没有已注册的 model，无法创建 optimizer")
        params = []
        for m in models:
            params += list(m.parameters())
        if isinstance(opt_type, str):
            cls = getattr(torch.optim, opt_type, None)
            if cls is None:
                raise ValueError(f"torch.optim 中找不到 '{opt_type}'")
        else:
            cls = opt_type
        optimizer = cls(params, **kwargs)
        self.add_optimizer(optimizer, space=space)
        self._init_specs['optimizer'].append({
            'space': space,
            'type': cls.__name__,
            'kwargs': dict(kwargs),
            'param_count': sum(p.numel() for p in params),
        })
        self.emit('new_optimizer', data=self._init_specs['optimizer'][-1])
        return self
    
    def new_criterion(
        self,
        space: str,
        cri_type: Union[str, Type[Callable], Callable],
        **kwargs,
    ) -> 'Engine':
        if isinstance(cri_type, str):
            cls = getattr(torch.nn, cri_type, None)
            if cls is None:
                raise ValueError(f"torch.nn 中找不到 '{cri_type}'")
            criterion = cls(**kwargs)
            type_name = cls.__name__
        elif isinstance(cri_type, type):
            criterion = cri_type(**kwargs)
            type_name = cri_type.__name__
        else:
            criterion = cri_type
            type_name = cri_type.__class__.__name__
        self.add_criterion(criterion, space=space)
        self._init_specs['criterion'].append({
            'space': space,
            'type': type_name,
            'kwargs': dict(kwargs),
        })
        self.emit('new_criterion', data=self._init_specs['criterion'][-1])
        return self
    

    def to(self, device: str) -> 'Engine':
        self.device = torch.device(device)
        for space in self._moc:
            for i in range(len(self._moc[space]['model'])):
                self._moc[space]['model'][i] = self._moc[space]['model'][i].to(self.device)
        return self

    def zero_grad(self, space: Union[str, None] = None) -> 'Engine':
        self.emit('start_zero_grad', data={'space': space})
        target_spaces = [space] if space is not None else list(self._moc.keys())
        for sp in target_spaces:
            if sp not in self._moc: continue
            for opt in self._moc[sp]['optimizer']:
                self.emit('before_zero_grad', data={'optimizer': opt, 'space': sp})
                opt.zero_grad()
                self.emit('after_zero_grad', data={'optimizer': opt, 'space': sp})
        self.emit('end_zero_grad', data={'space': space})
        return self

    def step(self, space: Union[str, None] = None) -> 'Engine':
        self.emit('start_opt_step', data={'space': space})
        target_spaces = [space] if space is not None else list(self._moc.keys())
        for sp in target_spaces:
            if sp not in self._moc: continue

            if self.scaler.is_enabled():
                for opt in self._moc[sp]['optimizer']:
                    self.scaler.unscale_(opt)

            if self.clip_grad:
                for model in self._moc[sp]['model']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_max_norm)

            for opt in self._moc[sp]['optimizer']:
                self.emit('before_opt_step', data={'optimizer': opt, 'space': sp})
                if self.scaler.is_enabled():
                    self.scaler.step(opt)
                else:
                    opt.step()
                self.emit('after_opt_step', data={'optimizer': opt, 'space': sp})

        if self.scaler.is_enabled():
            self.scaler.update()

        self.emit('end_opt_step', data={'space': space})
        return self

    def step_if_ready(self, space: Optional[str] = None) -> bool:
        if not self.is_training:
            return False
        if self.step_micro % self.accumulate_grad_batches != 0:
            return False
        self.step(space=space)
        self.zero_grad(space=space)
        return True

    def process_batch_data(self, batch: Any):
        self.emit('start_process_batch_data', data=batch)
        self.data, self.target = process_batch_data(batch, self.device)
        self.emit('end_process_batch_data')

    def update_loss(self, loss: Union[torch.Tensor, dict], space: str = 'default') -> 'Engine':
        if isinstance(loss, dict):
            for sp, res in loss.items():
                losses = res.get('losses', [])
                if not losses: continue
                total = sum(losses) / self.accumulate_grad_batches
                self.emit('before_update_loss', data={'loss': total, 'space': sp})
                self.scaler.scale(total).backward()
                self.emit('after_update_loss', data={'loss': total, 'space': sp})
        else:
            loss = loss / self.accumulate_grad_batches
            self.emit('before_update_loss', data={'loss': loss, 'space': space})
            self.scaler.scale(loss).backward()
            self.emit('after_update_loss', data={'loss': loss, 'space': space})
        return self

    def forward_pass(self, space: Union[str, None] = None) -> dict:
        target_spaces = [space] if space is not None else list(self._moc.keys())
        self.emit('start_forward_pass', data={'space': target_spaces})

        results = {}
        dtype = torch.float16 if self.precision == '16' else (torch.bfloat16 if self.precision == 'bf16' else torch.float32)

        grad_ctx = nullcontext() if self.is_training else torch.no_grad()
        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=dtype,
            enabled=self.precision in ['16', 'bf16']
        )

        with grad_ctx, autocast_ctx:
            for sp in target_spaces:
                if sp not in self._moc: continue

                x = self.data
                for model in self._moc[sp]['model']:
                    self.emit('before_forward_pass', data={'model': model, 'space': sp})
                    x = model(x)
                    self.emit('after_forward_pass', data={'model': model, 'space': sp, 'output': x})

                losses = []
                for criterion in self._moc[sp]['criterion']:
                    self.emit('before_criterion', data={'criterion': criterion, 'space': sp})
                    loss = criterion(x, self.target)
                    losses.append(loss)
                    self.emit('after_criterion', data={'criterion': criterion, 'space': sp, 'loss': loss})

                results[sp] = {'output': x, 'losses': losses}

        self.emit('end_forward_pass', data={'space': space, 'results': results})
        return results

    def auto_update(self, space: Union[str, None] = None) -> dict:
        self.emit('start_auto_update', data={'space': space})

        results = self.forward_pass(space=space)

        if not self.is_training:
            self.emit('end_auto_update', data={'space': space, 'results': results})
            return results

        loss_payload = {
            sp: {'losses': res['losses']}
            for sp, res in results.items()
            if (space is None or sp == space) and res.get('losses')
        }
        if loss_payload:
            self.update_loss(loss_payload)

        if self.step_micro % self.accumulate_grad_batches == 0:
            self.step(space=space)
            self.zero_grad(space=space)

        self.emit('end_auto_update', data={'space': space, 'results': results})
        return results

    def fit_once(self, dataloader: torch.utils.data.DataLoader) -> Generator:
        self.emit('start_epoch', data=dataloader)
        self.step_global_in_batch = 0
        for step, batch in enumerate(dataloader):
            self.emit('before_step')
            self.process_batch_data(batch)
            self.step_micro_in_batch = step
            if self.is_training:
                self.step_micro += 1
            self.emit('before_yield')
            yield
            self.emit('after_yield')

            if self.is_training and self.step_micro % self.accumulate_grad_batches == 0:
                self.step_global += 1
                self.step_global_in_batch += 1
            self.step_if_ready()

            self.emit('after_step')
        self.emit('end_epoch')

    def get_checkpoint_state(self) -> Dict[str, Any]:
        '''
        Create a checkpoint state dictionary.

        Returns:
            Dict[str, Any]: The checkpoint state containing engine state and all models/optimizers.
        '''
        spaces_state = {}
        for space_name, space_data in self._moc.items():
            spaces_state[space_name] = {
                'models': [m.state_dict() for m in space_data['model']],
                'optimizers': [o.state_dict() for o in space_data['optimizer']],
            }
        
        return {
            'epoch': self.epoch,
            'step_global': self.step_global,
            'step_micro': self.step_micro,
            'spaces': spaces_state,
        }
    
    def save_checkpoint(self, filename: Optional[str] = None) -> Optional[Any]:
        '''
        Save a checkpoint using the recorder.

        Args:
            filename (Optional[str]): Custom filename for the checkpoint.

        Returns:
            Optional[Path]: Path to the saved checkpoint, or None if recorder is not enabled.
        '''
        if not self._recorder.enable:
            return None
        self._recorder.flush()
        state = self.get_checkpoint_state()
        return self._recorder.save_checkpoint(state, filename)
    
    def flush(self) -> 'Engine':
        if self._recorder.enable: self._recorder.flush()
        return self
    
    def _close(self) -> 'Engine':
        '''
        Close the engine and flush all pending recorder tasks.
        '''
        self.flush()
        if self._recorder.enable: self._recorder.close()
        return self