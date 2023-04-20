import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def createLossAndOptimizer(net, learning_rate, scheduler_step, scheduler_gamma):
    loss = LossFunc()
    # optimizer = optim.Adam([{'params': net.parameters(), 'lr':learning_rate}], lr = learning_rate, weight_decay=5e-4)
    optimizer = optim.Adam([{'params': net.parameters(), 'lr': learning_rate}], lr=learning_rate, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    return loss, optimizer, scheduler

class LossFunc(torch.nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()

    def mse_loss(self, score, label):
        score = torch.squeeze(score)
        return torch.mean((score - label) ** 2)

    def forward(self, score, label):
        mse = self.mse_loss(score, label)

        return mse
