import math
import time
import torch
import torch.nn as nn
from flow import *
import os

class CDFlow(nn.Module):
    def __init__(self):
        super(CDFlow, self).__init__()
        self.glow = Glow(3, 8, 6, affine=True, conv_lu=True)

    def coordinate_transform(self, x_hat, rev=False, cd_map=False):
        if not rev:
            log_p, logdet, x_hat = self.glow(x_hat)
            return log_p, logdet, x_hat

        else:
            x_hat = self.glow.reverse(x_hat, cd_map=cd_map)

            return x_hat

    def forward(self, x, y):
        log_p_x, logdet_x, x_hat = self.coordinate_transform(x, rev=False)
        log_p_y, logdet_y, y_hat = self.coordinate_transform(y, rev=False)

        x_hat_1, y_hat_1 = x_hat[0].view(x_hat[0].shape[0], -1), y_hat[0].view(x_hat[0].shape[0], -1)
        x_hat_2, y_hat_2 = x_hat[1].view(x_hat[1].shape[0], -1), y_hat[1].view(x_hat[1].shape[0], -1)
        x_hat_3, y_hat_3 = x_hat[2].view(x_hat[2].shape[0], -1), y_hat[2].view(x_hat[2].shape[0], -1)
        x_hat_4, y_hat_4 = x_hat[3].view(x_hat[3].shape[0], -1), y_hat[3].view(x_hat[3].shape[0], -1)
        x_hat_5, y_hat_5 = x_hat[4].view(x_hat[4].shape[0], -1), y_hat[4].view(x_hat[4].shape[0], -1)
        x_hat_6, y_hat_6 = x_hat[5].view(x_hat[5].shape[0], -1), y_hat[5].view(x_hat[5].shape[0], -1)

        x_cat_65 = torch.cat((x_hat_6, x_hat_5), dim=1)
        y_cat_65 = torch.cat((y_hat_6, y_hat_5), dim=1)
        x_cat_654 = torch.cat((x_hat_6, x_hat_5, x_hat_4), dim=1)
        y_cat_654 = torch.cat((y_hat_6, y_hat_5, y_hat_4), dim=1)
        x_cat_6543 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3), dim=1)
        y_cat_6543 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3), dim=1)
        x_cat_65432 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3, x_hat_2), dim=1)
        y_cat_65432 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3, y_hat_2), dim=1)
        x_cat_654321 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3, x_hat_2, x_hat_1), dim=1)
        y_cat_654321 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3, y_hat_2, y_hat_1), dim=1)

        mse6 = (x_hat_6 - y_hat_6).view(x_hat_6.shape[0], -1)
        mse6 = mse6.unsqueeze(1)
        mse6 = torch.sqrt(1e-8 + torch.matmul(mse6, mse6.transpose(dim0=-2, dim1=-1))/mse6.shape[2])
        mse6 = mse6.squeeze(2)

        mse65 = (x_cat_65 - y_cat_65).view(x_cat_65.shape[0], -1)
        mse65 = mse65.unsqueeze(1)
        mse65 = torch.sqrt(1e-8 + torch.matmul(mse65, mse65.transpose(dim0=-2, dim1=-1))/mse65.shape[2])
        mse65 = mse65.squeeze(2)

        mse654 = (x_cat_654 - y_cat_654).view(x_cat_654.shape[0], -1)
        mse654 = mse654.unsqueeze(1)
        mse654 = torch.sqrt(1e-8 + torch.matmul(mse654, mse654.transpose(dim0=-2, dim1=-1))/mse654.shape[2])
        mse654 = mse654.squeeze(2)

        mse6543 = (x_cat_6543 - y_cat_6543).view(x_cat_6543.shape[0], -1)
        mse6543 = mse6543.unsqueeze(1)
        mse6543 = torch.sqrt(1e-8 + torch.matmul(mse6543, mse6543.transpose(dim0=-2, dim1=-1))/mse6543.shape[2])
        mse6543 = mse6543.squeeze(2)

        mse65432 = (x_cat_65432 - y_cat_65432).view(x_cat_65432.shape[0], -1)
        mse65432 = mse65432.unsqueeze(1)
        mse65432 = torch.sqrt(1e-8 + torch.matmul(mse65432, mse65432.transpose(dim0=-2, dim1=-1)) / mse65432.shape[2])
        mse65432 = mse65432.squeeze(2)

        mse654321 = (x_cat_654321 - y_cat_654321).view(x_cat_654321.shape[0], -1)
        mse654321 = mse654321.unsqueeze(1)
        mse654321 = torch.sqrt(1e-8 + torch.matmul(mse654321, mse654321.transpose(dim0=-2, dim1=-1)) / mse654321.shape[2])
        mse654321 = mse654321.squeeze(2)

        return mse654321, mse65432, mse6543, mse654, mse65, mse6, log_p_x, logdet_x, log_p_y, logdet_y

