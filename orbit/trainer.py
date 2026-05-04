import torch
from typing import Generator, Optional
from .engine import Engine


class Trainer(Engine):
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

    def run_loop(
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
