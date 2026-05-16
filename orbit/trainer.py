import os
import socket
import torch
import torch.multiprocessing as mp
from typing import Callable, Generator, Optional

from .engine import Engine


def _find_free_port() -> str:
    '''
    Locate a free TCP port on localhost. Used as the default rendezvous port
    when launching local DDP workers.
    '''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def _ddp_worker(local_rank: int, world_size: int, main_fn: Callable, backend: str) -> None:
    '''
    Entry point executed inside every spawned worker process.

    Sets up the rendezvous environment variables expected by torch.distributed
    and calls the user-provided main function. Defined at module top level so
    it is picklable across spawn-based start methods (Windows / macOS).

    Args:
        local_rank (int): Index of this process on the local node.
        world_size (int): Total number of processes participating.
        main_fn (Callable): User entry point.
        backend (str): Process group backend (informational; reserved).
    '''
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    main_fn()


class Trainer(Engine):
    @staticmethod
    def launch(
        main_fn: Callable[[], None],
        nprocs: Optional[int] = None,
        backend: str = 'nccl',
        master_addr: str = '127.0.0.1',
        master_port: Optional[str] = None,
    ) -> None:
        '''
        Launch `main_fn` across all available local GPUs using DDP.

        Behavior:
            - If running under torchrun (WORLD_SIZE > 1), runs `main_fn` directly.
            - If only one device is available, runs `main_fn` directly.
            - Otherwise spawns `nprocs` (default: torch.cuda.device_count())
              worker processes via torch.multiprocessing.

        Args:
            main_fn (Callable[[], None]): User entry point.
            nprocs (Optional[int]): Number of worker processes. Defaults to
                the count of visible CUDA devices.
            backend (str): Distributed backend (currently informational; the
                Engine reads this when constructing DistributedContext).
            master_addr (str): Rendezvous host. Defaults to '127.0.0.1'.
            master_port (Optional[str]): Rendezvous port. Defaults to a free
                port chosen at launch time.
        '''
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            main_fn()
            return

        n = nprocs if nprocs is not None else torch.cuda.device_count()
        if n is None or n <= 1:
            main_fn()
            return

        os.environ.setdefault('MASTER_ADDR', master_addr)
        os.environ.setdefault('MASTER_PORT', master_port if master_port is not None else _find_free_port())

        mp.spawn(_ddp_worker, nprocs=n, args=(n, main_fn, backend), join=True)

    def _loop(self, dataloader: torch.utils.data.DataLoader, mode: str) -> Generator:
        self.set_models_mode(mode)
        self.emit(f'start_{mode}_epoch', data={'epoch': self.epoch})
        yield from self.fit_once(dataloader)
        self.emit(f'end_{mode}_epoch', data={'epoch': self.epoch})

    def train(self, dataloader: torch.utils.data.DataLoader) -> Generator:
        yield from self._loop(dataloader, 'train')

    def eval(self, dataloader: torch.utils.data.DataLoader) -> Generator:
        yield from self._loop(dataloader, 'eval')

    def auto_train(self, dataloader: torch.utils.data.DataLoader, space: Optional[str] = None) -> None:
        '''
        Automatically runs the training loop for one epoch and performs parameter updates.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader for training data.
            space (Optional[str]): Target space to update. Default is None (all spaces).
        '''
        for _ in self.train(dataloader):
            self.auto_update(space=space)

    def auto_eval(self, dataloader: torch.utils.data.DataLoader, space: Optional[str] = None) -> None:
        '''
        Automatically runs the evaluation loop for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader for evaluation data.
            space (Optional[str]): Target space to forward pass. Default is None (all spaces).
        '''
        for _ in self.eval(dataloader):
            self.forward_pass(space=space)

    def epochs(
        self,
        max_epochs: int,
        start_epoch: int = 0
    ) -> Generator[int, None, None]:
        '''
        A generator that manages the outer training loop, emitting run events 
        and yielding the current epoch number.

        Args:
            max_epochs (int): The maximum number of epochs to run.
            start_epoch (int): The starting epoch number (default: 0).

        Yields:
            int: The current epoch number.
        '''
        self.emit('start_run', data={'max_epochs': max_epochs, 'start_epoch': start_epoch})
        self.is_run_finished = False
        self.epoch = start_epoch

        while not getattr(self, 'is_run_finished', False) and self.epoch < max_epochs:
            self.is_epoch_finished = False

            yield self.epoch

            self.step_schedules()
            self.epoch += 1

        self.emit('end_run')

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        self.emit('start_run', data={'start_epoch': self.epoch})

        self.is_run_finished = False

        while not getattr(self, 'is_run_finished', False):
            self.is_epoch_finished = False

            self.auto_train(train_loader)

            if val_loader is not None:
                self.auto_eval(val_loader)

            self.step_schedules()
            self.epoch += 1

        self.emit('end_run')
