import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHardQuantize(nn.Module):
    def __init__(self, n_embed, embed_dim, embed_init,
                 commitment, anneal, sigma_start, sigma_max, sigma_period, sigma_scale):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.commitment = commitment
        self.anneal = anneal
        self.sigma_start = sigma_start
        self.sigma_max = sigma_max
        self.sigma_period = sigma_period
        self.sigma_scale = sigma_scale

        self.register_buffer('embed', embed_init)
        self.register_buffer('steps', torch.ones(1))

    def _linear_annealing(self):
        sigma = self.sigma_scale * torch.div(self.steps, self.sigma_period, rounding_mode='trunc') + self.sigma_start
        sigma = torch.clamp(sigma, 0, self.sigma_max)
        self.steps += 1
        return sigma

    def _anneal(self):
        match self.anneal:
            case 'linear':
                return self._linear_annealing()
            case _:
                raise NotImplementedError

    def forward(self, x):
        if self.training:
            sigma = self._anneal()
        else:
            sigma = 1e10 * torch.ones(1, device=x.device)

        x = x.view(x.size(0), -1, self.embed_dim)
        flatten = x.view(-1, self.embed_dim)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        soft_assign = F.softmax(-sigma * dist, dim=1)
        likelihoods = torch.mean(soft_assign, dim=0)
        embed_ind = soft_assign.argmax(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        soft_assign = torch.matmul(soft_assign, self.embed.transpose(0, 1))
        soft_assign = soft_assign.view(*x.shape)

        quantize = soft_assign + (quantize - soft_assign).detach()

        # add extra loss as kl divergence between likelihood and uniform distribution
        uniform_prob = torch.ones_like(likelihoods) / self.n_embed
        quant_loss = F.kl_div(torch.log(likelihoods.detach() + 1e-10), uniform_prob, reduction='batchmean') * self.commitment
        return quantize, {'quant_loss': quant_loss,
                          'embed': self.embed.transpose(0, 1).clone().detach(),
                          'likelihoods': likelihoods.clone().detach(),
                          'sigma': sigma.item()}


class LearnedQuantize(nn.Module):
    def __init__(self, n_embed, embed_dim, embed_init, normalize,
                 commitment, anneal, sigma_start, sigma_max, sigma_period, sigma_scale):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embed = nn.Parameter(embed_init, requires_grad=True)
        self.normalize = normalize

        self.commitment = commitment
        self.anneal = anneal
        self.sigma_start = sigma_start
        self.sigma_max = sigma_max
        self.sigma_period = sigma_period
        self.sigma_scale = sigma_scale

        self.register_buffer('steps', torch.ones(1))

    def _linear_annealing(self):
        sigma = self.sigma_scale * torch.div(self.steps, self.sigma_period, rounding_mode='trunc') + self.sigma_start
        sigma = torch.clamp(sigma, 0, self.sigma_max)
        self.steps += 1
        return sigma

    def _anneal(self):
        match self.anneal:
            case 'linear':
                return self._linear_annealing()
            case _:
                raise NotImplementedError

    def forward(self, x):
        if self.training:
            sigma = self._anneal()
        else:
            sigma = 1e10

        sigma = self._anneal()

        x = x.view(x.size(0), -1, self.embed_dim)
        flatten = x.view(-1, self.embed_dim)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        soft_assign = F.softmax(-sigma * dist, dim=1)
        likelihoods = torch.mean(soft_assign, dim=0)
        embed_ind = soft_assign.argmax(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        soft_assign = torch.matmul(soft_assign, self.embed.transpose(0, 1))
        soft_assign = soft_assign.view(*x.shape)

        quantize = soft_assign + (quantize - soft_assign).detach()

        if self.normalize:
            norm_factor = torch.sqrt(
                torch.matmul(likelihoods,
                            torch.square(torch.view_as_complex(self.embed.transpose(0, 1)).abs()))
            )
            quantize = quantize / norm_factor

        # add extra loss as kl divergence between likelihood and uniform distribution
        uniform_prob = torch.ones_like(likelihoods) / self.n_embed
        quant_loss = F.kl_div(torch.log(likelihoods.detach() + 1e-10), uniform_prob, reduction='batchmean') * self.commitment
        return quantize, {'quant_loss': quant_loss,
                          'embed': self.embed.transpose(0, 1).clone().detach(),
                          'likelihoods': likelihoods.clone().detach(),
                          'sigma': sigma.item()}
