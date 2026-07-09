import torch
import torch.nn as nn

#Sangeetha Kamalaksha Kini 392545 

#Reparameterize sample 
def reparameterize(m: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    ep = torch.randn_like(std)
    z = m + ep * std
    return z


def sample_prior(batch_size: int, latent_dim: int, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    return torch.randn(batch_size, latent_dim, device=device)

#KL Divergence between learned and prior distribution
def kl_divergence(m: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - m.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")


def interpolate(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    z1 = z1.view(-1)
    z2 = z2.view(-1)
    alphas = torch.linspace(0, 1, steps, device=z1.device).unsqueeze(1)  # (steps, 1)
    z_interp = (1 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)
    return z_interp  #Needs to be decoded to get the sentences from the latent space. This should be done in the decoder.py file

class LatentSpace(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return reparameterize(mu, logvar)
 
    @staticmethod
    def kl(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return kl_divergence(mu, logvar, reduction=reduction)
 
    @staticmethod
    def sample(batch_size: int, latent_dim: int, device=None) -> torch.Tensor:
        return sample_prior(batch_size, latent_dim, device=device)

#Only for testing the funtions in this file. Remove after integrating with main code
'''if __name__ == "__main__":

    batch_size, latent_dim = 4, 16

    m = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim) * 0.1  # small variance for sane demo

    z = reparameterize(m, logvar)
    print("z shape:", z.shape)  

    kl = kl_divergence(m, logvar)
    print("KL mean:", kl.item())

    z_prior = sample_prior(batch_size, latent_dim)
    print("Prior sample shape:", z_prior.shape)

    z_path = interpolate(z[0], z[1], steps=5)
    print("Interpolation shape:", z_path.shape) '''

#Sangeetha Kamalaksha Kini 392545 
