import shutil
import random
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def copy_codes(trainpath1,trainpath2,trainpath3,trainpath4, path1,path2,path3,path4):
    shutil.copyfile(trainpath1, path1)
    shutil.copyfile(trainpath2, path2)
    shutil.copyfile(trainpath3, path3)
    shutil.copyfile(trainpath4, path4)