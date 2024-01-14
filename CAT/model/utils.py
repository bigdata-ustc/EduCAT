import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        actions = hard_sample(logits)
        return actions

class StraightThrough:
    def __init__(self, state_dim, action_dim, lr,  config):
        self.lr = lr
        device = config['device']
        self.betas = config['betas']
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=self.betas)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()