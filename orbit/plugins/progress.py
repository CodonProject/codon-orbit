import torch
from typing import Any, Optional
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    ProgressColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from ..event import Event

class RichProgressBar:
    '''
    A progress bar plugin using the rich library for orbit engine.

    This plugin provides a visual progress bar in the terminal, showing training 
    and evaluation progress along with real-time metrics.

    Attributes:
        progress (Progress): The rich progress object.
        live (Live): The rich live display object.
        main_task_id (Optional[int]): ID for the overall progress task.
        epoch_task_id (Optional[int]): ID for the current epoch progress task.
    '''

    def __init__(self) -> None:
        '''
        Initializes the RichProgressBar plugin.
        '''
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn('|'),
            TimeRemainingColumn(),
            expand=True
        )
        self.live: Optional[Live] = None
        self.main_task_id: Optional[int] = None
        self.epoch_task_id: Optional[int] = None
        self.current_mode: str = 'train'

    def setup(self, engine: Any) -> None:
        '''
        Setup the plugin by initializing the live display.

        Args:
            engine (Any): The orbit engine instance.
        '''
        self.live = Live(self.progress, refresh_per_second=10)
        self.live.start()

    def teardown(self, engine: Any) -> None:
        '''
        Teardown the plugin by stopping the live display.

        Args:
            engine (Any): The orbit engine instance.
        '''
        if self.live:
            self.live.stop()

    def on_start_run(self, event: Event) -> None:
        '''
        Handles the start of the entire run.

        Args:
            event (Event): The start_run event.
        '''
        data = event.data or {}
        max_epochs = data.get('max_epochs', None)
        start_epoch = data.get('start_epoch', 0)
        self.main_task_id = self.progress.add_task(
            '[bold blue]Overall Progress',
            total=max_epochs,
            completed=start_epoch
        )

    def on_start_train_epoch(self, event: Event) -> None:
        '''
        Sets mode to training.

        Args:
            event (Event): The start_train_epoch event.
        '''
        self.current_mode = 'train'

    def on_start_eval_epoch(self, event: Event) -> None:
        '''
        Sets mode to evaluation.

        Args:
            event (Event): The start_eval_epoch event.
        '''
        self.current_mode = 'eval'

    def on_start_epoch(self, event: Event) -> None:
        '''
        Handles the start of a single epoch (train or eval).

        Args:
            event (Event): The start_epoch event.
        '''
        dataloader = event.data
        total_steps = len(dataloader)
        epoch_num = event.engine.epoch
        
        desc = f'[green]Epoch {epoch_num} [{self.current_mode}]'
        if self.epoch_task_id is not None:
            self.progress.remove_task(self.epoch_task_id)
            
        self.epoch_task_id = self.progress.add_task(desc, total=total_steps)

    def on_after_step(self, event: Event) -> None:
        '''
        Updates the progress bar after each step.

        Args:
            event (Event): The after_step event.
        '''
        if self.epoch_task_id is not None:
            engine = event.engine
            loss_val = 'N/A'
            if engine.loss is not None:
                if isinstance(engine.loss, torch.Tensor):
                    loss_val = f'{engine.loss.item():.4f}'
                else:
                    loss_val = f'{engine.loss:.4f}'
                    
            lr_val = 'N/A'
            try:
                # Attempt to get learning rate from the first optimizer in 'main' space
                opt = engine._moc['main']['optimizer'][0]
                lr_val = f"{opt.param_groups[0]['lr']:.2e}"
            except Exception:
                pass
            
            self.progress.update(
                self.epoch_task_id,
                advance=1,
                description=f'[green]Epoch {engine.epoch} [{self.current_mode}] LR: {lr_val} Loss: {loss_val}'
            )

    def on_end_epoch(self, event: Event) -> None:
        '''
        Updates the main progress bar when an epoch ends.

        Args:
            event (Event): The end_epoch event.
        '''
        if self.current_mode == 'train' and self.main_task_id is not None:
            self.progress.advance(self.main_task_id)

    def on_end_run(self, event: Event) -> None:
        '''
        Cleanup when the run ends.

        Args:
            event (Event): The end_run event.
        '''
        if self.live:
            self.live.stop()
