import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

from backbone.resnet import *
from utils.data_utils import *
from utils.loss_utils import *
from trainer import *
from utils.eval_utils import *


# Main training script
def main():
    # Hyperparameters
    batch_size = 512
    epochs = 100
    learning_rate = 3e-4
    temperature = 0.5
    projection_dim = 128

    os.makedirs("./tsne", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load CIFAR-10 with contrastive augmentation
    train_transform = ContrastiveTransformations(
        get_simclr_augmentation(), n_views=2
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )
    # Load CIFAR-10 for evals:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset_eval = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=test_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader_eval = DataLoader(
        train_dataset_eval, batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )


    # Model, loss, and optimizer
    model = ResNetEncoder(out_dim=projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Training loop
    print('Starting contrastive pre-training...')
    for epoch in range(1, epochs + 1):
        train_loss = train_contrastive(
            model, train_loader, optimizer, criterion, device, epoch
        )
        if epoch % 5 == 0:
            h_2d, z1_2d, z2_2d, labels = visualize_tsne_comparison(model, test_loader, device,
                                                             save_path =f"tsne/e{epoch}_plot")
        scheduler.step()

    # Save the pre-trained model
    torch.save(model.state_dict(), 'simclr_cifar10.pth')
    print('Pre-training complete! Model saved.')

    # Linear evaluation
    print('\nStarting linear evaluation...')
    final_accuracy = linear_eval(
        model, train_loader_eval, test_loader, device, epochs=100
    )
    print(f'\nFinal Test Accuracy: {final_accuracy:.2f}%')


if __name__ == '__main__':
    main()