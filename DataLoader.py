import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision


class CD_128(Dataset):
    # without norm
    def __init__(self, jnd_info, root_dir, test=False):
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        self.test = test
        random.seed(20)
        torch.random.manual_seed(20)
        if test == False:
            self.trans_org = transforms.Compose([
                transforms.Resize(1024),
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                transforms.RandomCrop(768),
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.Resize(1024),
                transforms.CenterCrop(1024),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        ref1 = self.trans_org(ref)

        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))

        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        test1 = self.trans_org(test)

        return ref1, test1, gt
