import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN
import torchvision.transforms as transforms
from model import Generator, Discriminator, ResidualBlock
from utils import cycle_consistency_loss, identity_loss
import itertools
import os

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
mnist_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

# Load SVHN dataset
svhn_dataset = SVHN(root='./data', split='train', transform=transform, download=True)

# Define data loaders
batch_size = 64
mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True)

# Define input and output channels
input_channels = 3  # For SVHN dataset
output_channels = 1  # For MNIST dataset

# Define generators
G_MNIST2SVHN = Generator(input_channels, output_channels)
G_SVHN2MNIST = Generator(output_channels, input_channels)

# Define discriminators
D_MNIST = Discriminator(input_channels)
D_SVHN = Discriminator(output_channels)

# Initialize optimizers
optimizer_G = optim.Adam(itertools.chain(G_MNIST2SVHN.parameters(), G_SVHN2MNIST.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_MNIST = optim.Adam(D_MNIST.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_SVHN = optim.Adam(D_SVHN.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define loss functions
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (mnist_images, svhn_images) in enumerate(zip(mnist_loader, svhn_loader)):
        # Clear gradients
        optimizer_G.zero_grad()
        optimizer_D_MNIST.zero_grad()
        optimizer_D_SVHN.zero_grad()

        # Forward pass
        mnist2svhn = G_MNIST2SVHN(mnist_images)
        svhn2mnist = G_SVHN2MNIST(svhn_images)
        mnist2svhn_cycle = G_SVHN2MNIST(mnist2svhn)
        svhn2mnist_cycle = G_MNIST2SVHN(svhn2mnist)

        # Compute losses
        cycle_loss_MNIST2SVHN = criterion_cycle(mnist2svhn_cycle, mnist_images)
        cycle_loss_SVHN2MNIST = criterion_cycle(svhn2mnist_cycle, svhn_images)
        identity_loss_MNIST2SVHN = criterion_identity(G_SVHN2MNIST(svhn_images), svhn_images)
        identity_loss_SVHN2MNIST = criterion_identity(G_MNIST2SVHN(mnist_images), mnist_images)
        total_cycle_loss = cycle_loss_MNIST2SVHN + cycle_loss_SVHN2MNIST
        total_identity_loss = identity_loss_MNIST2SVHN + identity_loss_SVHN2MNIST

        # Compute generator and discriminator losses
        generator_loss = total_cycle_loss + 0.5 * total_identity_loss
        discriminator_loss_MNIST = D_MNIST(mnist_images).mean() - D_MNIST(G_SVHN2MNIST(svhn_images)).mean()
        discriminator_loss_SVHN = D_SVHN(svhn_images).mean() - D_SVHN(G_MNIST2SVHN(mnist_images)).mean()

        # Backward pass
        generator_loss.backward(retain_graph=True)
        discriminator_loss_MNIST.backward(retain_graph=True)
        discriminator_loss_SVHN.backward()

        # Update weights
        optimizer_G.step()
        optimizer_D_MNIST.step()
        optimizer_D_SVHN.step()

        if epoch % 10 == 0:
            torch.save(G_MNIST2SVHN.state_dict(), os.path.join(save_dir, f'G_MNIST2SVHN_epoch_{epoch}.pth'))
            torch.save(G_SVHN2MNIST.state_dict(), os.path.join(save_dir, f'G_SVHN2MNIST_epoch_{epoch}.pth'))

        # Print loss
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(mnist_loader)}], '
                  f'Generator Loss: {generator_loss.item():.4f}, '
                  f'Discriminator Loss MNIST: {discriminator_loss_MNIST.item():.4f}, '
                  f'Discriminator Loss SVHN: {discriminator_loss_SVHN.item():.4f}')
