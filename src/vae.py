import torch
import torch.nn as nn

class BasicVAE(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, 2); self.logvar = nn.Linear(64, 2)
        self.dec = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, input_dim))
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
    def forward(self, x):
        h = self.enc(x)
        return self.dec(self.reparameterize(self.mu(h), self.logvar(h))), self.mu(h), self.logvar(h)

class HybridBetaVAE(nn.Module):
    def __init__(self, beta=4.0):
        super().__init__()
        self.beta = beta
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU())
        self.flat_dim = 119808
        self.fc_fusion = nn.Linear(self.flat_dim + 768, 128)
        self.mu = nn.Linear(128, 64); self.logvar = nn.Linear(128, 64)
        self.dec_input = nn.Linear(64, self.flat_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (32, 16, 234)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, audio, text):
        a = self.cnn(audio).flatten(1)
        fused = torch.cat([a, text], dim=1)
        h = torch.relu(self.fc_fusion(fused))
        mu, logvar = self.mu(h), self.logvar(h)
        return self.dec(self.dec_input(mu + torch.randn_like(mu)*torch.exp(0.5*logvar))), mu, logvar
