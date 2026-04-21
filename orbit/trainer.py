import torch
from typing import Generator, Optional
from .engine import Engine


class Trainer(Engine):
    def __init__(self, max_epochs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.max_epochs = max_epochs

    def _loop(self, dataloader: torch.utils.data.DataLoader, mode: str) -> Generator:
        self.set_models_mode(mode)
        self.emit(f'start_{mode}_epoch', data={'epoch': self.epoch})
        yield from self.fit_once(dataloader)
        self.emit(f'end_{mode}_epoch', data={'epoch': self.epoch})

    def train(self, dataloader: torch.utils.data.DataLoader) -> Generator:
        yield from self._loop(dataloader, 'train')

    def eval(self, dataloader: torch.utils.data.DataLoader) -> Generator:
        yield from self._loop(dataloader, 'eval')

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        epochs = max_epochs if max_epochs is not None else self.max_epochs
        self.emit('start_run', data={'max_epochs': epochs})

        for ep in range(epochs):
            self.epoch = ep

            for _ in self.train(train_loader):
                self.auto_update()

            if val_loader is None: continue
            for _ in self.eval(val_loader):
                self.forward_pass()

        self.emit('end_run')