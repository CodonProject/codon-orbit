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
        flatten (nn.Flatten): Flattens the 2D image into 1D vector.
        fc1 (nn.Linear): First linear layer.
        activation (nn.SiLU): SiLU activation function.
        fc2 (nn.Linear): Output linear layer.
    '''
    def __init__(self, in_features: int = 784, hidden_dim: int = 128, out_features: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        '''
        x = self.flatten(input_tensor)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def main():
    # 1. Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./mnist', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 2. Build model
    model = SiluMLP()

    # 3. Setup Trainer
    trainer = Trainer(precision='32')
    
    trainer.add_model(model, space='main')
    trainer.new_optimizer(space='main', opt_type='Adam', lr=1e-3)
    trainer.new_schedule(space='main', sch_type='StepLR', step_size=2, gamma=0.1)
    trainer.new_criterion(space='main', cri_type='CrossEntropyLoss')
    
    # Move models to the correct device
    trainer.to(trainer.device)

    # Attach plugins
    trainer.attach(RichProgressBar())

    # 4. Start training (Generator run_loop mode)
    for ep in trainer.run_loop(max_epochs=5):
        trainer.auto_train(train_loader)
        trainer.auto_eval(val_loader)

if __name__ == '__main__':
    main()
