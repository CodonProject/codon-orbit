import os
from typing import Optional

import torch
import torch.distributed as dist

from .lifecycle import exit_manager


class DistributedContext:
    '''
    Encapsulates torch.distributed initialization and convenience helpers.

    Reads RANK / LOCAL_RANK / WORLD_SIZE from environment variables (set by
    `torchrun`). Initializes the process group exactly once and registers
    cleanup via the global ExitManager.
    '''

    _instance: Optional['DistributedContext'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, backend: str = 'nccl'):
        if getattr(self, '_initialized', False):
            return

        self._backend = backend
        self._world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self._rank = int(os.environ.get('RANK', '0'))
        self._local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        self._enabled = self._world_size > 1
        self._device: torch.device

        if self._enabled:
            if backend == 'nccl' and not torch.cuda.is_available():
                raise RuntimeError(
                    "DistributedContext: backend='nccl' requires CUDA, "
                    "fall back to backend='gloo' for CPU-only environments."
                )

            if torch.cuda.is_available():
                torch.cuda.set_device(self._local_rank)
                self._device = torch.device(f'cuda:{self._local_rank}')
            else:
                self._device = torch.device('cpu')

            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
                exit_manager.register(self.destroy)
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._initialized = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def barrier(self) -> None:
        if self._enabled and dist.is_initialized():
            dist.barrier()

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        In-place mean all-reduce. Returns the same tensor for chaining.
        Falls back to a no-op when distributed mode is disabled.
        '''
        if self._enabled and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(self._world_size)
        return tensor

    def destroy(self) -> None:
        if self._enabled and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
