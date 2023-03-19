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
        self.eps = 1e-6
        self.weight1 = 0.5
        self.weight2 = 1.5
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)  ## kernel_size = 4

    def spatial_loss(self, x, y):
        x_mean = torch.mean(x, 1, keepdim=True)  ## If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
        y_mean = torch.mean(y, 1, keepdim=True)
        org_pool = self.pool(x_mean)  ## average pooling
        enhance_pool = self.pool(y_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return self.weight1*torch.mean(E)
        
    def color_loss(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = (mr - mg) * (mr - mg)
        Drb = (mr - mb) * (mr - mb)
        Dgb = (mb - mg) * (mb - mg)
        k = Drg + Drb + Dgb
        return self.weight1*torch.mean(k)

    def Charbonnier_Loss(self, y, y_rev):
        distance = torch.sqrt((y - y_rev) ** 2 + self.eps)
        distance = distance.reshape(y.shape[0], -1)
        loss = torch.mean(distance)
        return loss

    def total_var(self, x, weight):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = ((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]) ** 2).sum()
        w_tv = ((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]) ** 2).sum()
        return weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def stress_loss(self, delta_e, delta_v):
        fcv = torch.sum(delta_e * delta_e) / (torch.sum(delta_e * delta_v)+ self.eps)
        stress = 100 * torch.sqrt(torch.sum((delta_e - fcv * delta_v) * (delta_e - fcv * delta_v)) / (fcv * fcv * torch.sum(delta_v * delta_v) + self.eps))
        return torch.mean(stress)

    def mse_loss(self, score, label):
        score = torch.squeeze(score)
        return torch.mean((score - label) ** 2)

    def forward(self, score, label):
        l_stress = self.mse_loss(score, label)

        return l_stress
