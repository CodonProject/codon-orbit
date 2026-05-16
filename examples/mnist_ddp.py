'''
Multi-GPU MNIST training example using DDP.

Zero-boilerplate launch:
    python examples/mnist_ddp.py
The script auto-detects all visible CUDA devices and spawns one worker per
GPU. Falls back to single-process execution on a single device or CPU.

Also compatible with torchrun:
    torchrun --nproc_per_node=NUM_GPUS examples/mnist_ddp.py
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from orbit.trainer import Trainer
from orbit.plugins import RichProgressBar


class SiluMLP(nn.Module):
    '''
    A simple Multi-Layer Perceptron (MLP) with SiLU activation for MNIST.

    Attributes:
        flatten (nn.Flatten): Flattens the 2D image into a 1D vector.
        fc1 (nn.Linear): First linear layer.
        activation (nn.SiLU): SiLU activation function.
        fc2 (nn.Linear): Output linear layer.
    '''

    def __init__(self, in_features: int = 784, hidden_dim: int = 128, out_features: int = 10) -> None:
        '''
        Initializes the SiluMLP block.

        Args:
            in_features (int): Size of each input sample.
            hidden_dim (int): Size of the hidden layer.
            out_features (int): Number of output classes.
        '''
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Class logits.
        '''
        x = self.flatten(input_tensor)
        x = self.fc1(x)
        x = self.activation(x)
        output = self.fc2(x)
        return output


def main() -> None:
    '''
    Build the trainer and run training. With `distributed='auto'` the engine
    detects `WORLD_SIZE` from the environment, moves models to the correct
    device, wraps them with DDP, and rebuilds dataloaders with a
    `DistributedSampler` on the fly. No explicit `to`, `wrap_ddp`, or
    `prepare_dataloader` calls are needed.
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./mnist', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = SiluMLP()

    trainer = Trainer(
        precision='32',
        distributed='auto',
        backend='nccl',
        sync_bn=False,
    )

    trainer.add_model(model, space='main')
    trainer.new_optimizer(space='main', opt_type='Adam', lr=1e-3)
    trainer.new_schedule(space='main', sch_type='StepLR', step_size=2, gamma=0.1)
    trainer.new_criterion(space='main', cri_type='CrossEntropyLoss')

    if trainer.is_main_process:
        trainer.attach(RichProgressBar())

    for _ in trainer.epochs(max_epochs=5):
        trainer.auto_train(train_loader)
        trainer.auto_eval(val_loader)


if __name__ == '__main__':
    Trainer.launch(main)
