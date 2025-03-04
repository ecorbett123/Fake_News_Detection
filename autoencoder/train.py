import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder, VAE
import numpy as np


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = ((recon_x - x)**2).mean()
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_autoencoder(embeddings, epochs=10, latent_dim=64):
    X_train_t = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(X_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = embeddings.shape[1]
    model = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] AE Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Autoencoder model saved as autoencoder_model.pth")


def train_vae(embeddings, epochs=10, latent_dim=64):
    X_train_t = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(X_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = embeddings.shape[1]
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in loader:
            x_batch = batch[0]
            x_recon, mu, logvar = model(x_batch)
            loss = vae_loss_function(x_recon, x_batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] VAE Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), 'vae_model.pth')
    print("VAE model saved as vae_model.pth")
