import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, input_dim=784):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def model_vae(train_loader, epochs=10, lr=1e-3, input_dim=784):
    model = VAE(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # Assuming data is a tuple like (inputs, labels) or just inputs
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, input_dim=input_dim)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}')
            
    return model

def vae_call(model, test_data):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_data:
            if isinstance(data, (list, tuple)):
                data = data[0]
            recon_batch, mu, logvar = model(data)
            predictions.append(recon_batch)
    return torch.cat(predictions)
