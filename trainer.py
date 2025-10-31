import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np


# Training function
def train_contrastive(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        # images is a list of 2 augmented views
        images = torch.cat(images, dim=0).to(device)  # [2*batch_size, 3, 32, 32]

        optimizer.zero_grad()

        # Forward pass
        _, z1, z2 = model(images)

        # Split into two views
        batch_size = z1.shape[0] // 2
        z_i = z1[:batch_size]
        z_j = z1[batch_size:]

        # Split into two views
        batch_size = z2.shape[0] // 2
        z_m = z2[:batch_size]
        z_n = z2[batch_size:]

        # Compute contrastive loss
        loss1 = criterion(z_i, z_j)
        loss2 = criterion(z_m, z_n)
        loss = loss1 - 0.05*loss2

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f},'
                  f'Loss1: {loss1.item()}, Loss2: {loss2.item()}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
    return avg_loss


# Linear evaluation protocol
def linear_eval(encoder, train_loader, test_loader, device, epochs=100):
    """
    Train a linear classifier on frozen features
    """
    encoder.eval()

    # Linear classifier
    classifier = nn.Linear(encoder.n_features, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features, _ = encoder(images)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0:
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    features, _ = encoder(images)
                    logits = classifier(features)
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            print(f'Linear Eval Epoch {epoch + 1}: Accuracy = {accuracy:.2f}%')

    return accuracy
