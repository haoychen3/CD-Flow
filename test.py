import time
from EMA import EMA
import torch
from torch.utils.data import DataLoader
from model import CDFlow
from DataLoader import CD_128
from coeff_func import *
import os
from loss import createLossAndOptimizer
from torch.autograd import Variable
import torchvision
import torch.autograd as autograd
from function import setup_seed, copy_codes


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size_test", type=int, default=20)
    parser.add_argument("--work_path", type=str, default='work_dir')
    parser.add_argument("--datapath", type=str, default='data')
    parser.add_argument("--dataset", type=str, default='/data3/image')
    parser.add_argument("--testset", type=str, default='test.csv')
    parser.add_argument("--test_aligned_path", type=str, default=None)
    parser.add_argument("--test_notaligned_path", type=str, default=None)

    return parser.parse_args()


def test(data_val_loader, net):
    dist = []
    y_true = []
    for i, data in enumerate(data_val_loader, 0):
        with torch.no_grad():
            x, y, gts = data
            y_val = gts.numpy()
            x, y, gts = \
                Variable(x).cuda(), \
                Variable(y).cuda(), \
                Variable(gts).cuda()
            score, _, _, _, _, _, _, _, _, _ = net(x, y)
            pred = (torch.squeeze(score)).cpu().detach().numpy().tolist()
            if isinstance(pred, list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)

    dist_np = np.array(dist)
    y_true_np = np.array(y_true).squeeze()
    stress = compute_stress(dist_np, y_true_np)
    _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist_np, y_true_np)

    return srocc_v, cc_v, stress, dist, y_true


config = parse_config()
path = config.datapath
work_path = config.work_path
testpath = config.testset
workspace = work_path + '/{}'.format(1)
testset = path + '/{}/'.format(1) + testpath
test_aligned_path = path + '/{}/test_aligned.csv'.format(1)
test_notaligned_path = path + '/{}/test_notaligned.csv'.format(1)
datadir = config.dataset
batch_size_test = config.batch_size_test

test_pairs = np.genfromtxt(open(testset, encoding='UTF-8-sig'), delimiter=',', dtype=str)
test_aligned_pairs = np.genfromtxt(open(test_aligned_path), delimiter=',', dtype=str)
test_notaligned_pairs = np.genfromtxt(open(test_notaligned_path), delimiter=',', dtype=str)

data_test = CD_128(test_pairs[:], root_dir=datadir, test=True)
test_aligned = CD_128(test_aligned_pairs[:], root_dir=datadir, test=True)
test_notaligned = CD_128(test_notaligned_pairs[:], root_dir=datadir, test=True)

data_test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=4)
data_test_aligned_loader = DataLoader(test_aligned, batch_size=batch_size_test, shuffle=False, pin_memory=True,
                                      num_workers=4)
data_test_notaligned_loader = DataLoader(test_notaligned, batch_size=batch_size_test, shuffle=False, pin_memory=True,
                                         num_workers=4)

print('#############################################################################')
print("Testing...")
print('#############################################################################')
device = torch.device("cuda")
pt = os.path.join(workspace, 'checkpoint_best', 'ModelParams_Best_val.pt')
checkpoint = torch.load(pt)
net = CDFlow().cuda()
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(checkpoint['state_dict'])
net.eval()
srocc_v1, cc_v1, stress1, dist1, y_true1 = test(data_test_loader, net)
print('All: plcc{}; srcc{}; stress{}'.format(cc_v1, srocc_v1, stress1))
srocc_v2, cc_v2, stress2, dist2, y_true2 = test(data_test_aligned_loader, net)
print('Pixel-wise aligned: plcc{}; srcc{}; stress{}'.format(cc_v2, srocc_v2, stress2))
srocc_v3, cc_v3, stress3, dist3, y_true3 = test(data_test_notaligned_loader, net)
print('Non-Pixel-wise aligned: plcc{}; srcc{}; stress{}'.format(cc_v3, srocc_v3, stress3))
