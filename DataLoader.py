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
                transforms.RandomHorizontalFlip(p=0.5),  ### need data augmentation?
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


class CD_128_trans(Dataset):
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
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.Resize(1024),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = './translation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png'
        seed = np.random.randint(2147483647)
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        ref1 = self.trans_org(ref)

        full_address_test = './translation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png'
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        test1 = self.trans_org(test)
        return ref1, test1, gt

class CD_128_rotate(Dataset):
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
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = './rotation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png'
        seed = np.random.randint(2147483647)
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        ref1 = self.trans_org(ref)

        full_address_test = './rotation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png'
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        test1 = self.trans_org(test)
        return ref1, test1, gt


class CD_128_dilate(Dataset):
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
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        full_address = './dilation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png'
        seed = np.random.randint(2147483647)
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        ref1 = self.trans_org(ref)

        full_address_test = './dilation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png'
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)
        torch.manual_seed(seed)
        test1 = self.trans_org(test)
        return ref1, test1, gt


class geo_trans(Dataset):
    # without norm
    def __init__(self, jnd_info, root_dir, test=True):
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        self.test = test
        if test == False:
            self.trans_1 = transforms.Compose([
                transforms.ToTensor(),

            ])
            self.trans_2 = transforms.Compose([
                transforms.ToTensor(),

            ])
        else:
            self.trans_1 = transforms.Compose([
                transforms.CenterCrop(896),
                transforms.ToTensor(),
                ])
            self.trans_2 = transforms.Compose([
                transforms.RandomCrop(896),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        print(idx)
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        ref = Image.open(full_address)
        ref1 = self.trans_1(ref)
        torchvision.utils.save_image(ref1, './translation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png')

        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test)
        test1 = self.trans_2(test)
        torchvision.utils.save_image(test1, './translation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png')

        return ref1, test1, gt

class geo_rotate(Dataset):
    # without norm
    def __init__(self, jnd_info, root_dir, test=True):
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        self.test = test
        if test == False:
            self.trans_1 = transforms.Compose([
                transforms.ToTensor(),

            ])
            self.trans_2 = transforms.Compose([
                transforms.ToTensor(),

            ])
        else:
            self.trans_1 = transforms.Compose([
                transforms.CenterCrop(896),
                transforms.ToTensor(),
                ])
            self.trans_2 = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.CenterCrop(896),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        print(idx)
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        ref = Image.open(full_address)
        ref1 = self.trans_1(ref)
        #torchvision.utils.save_image(ref1, full_address + str(idx) + '_1_r.png')
        torchvision.utils.save_image(ref1, './rotation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png')

        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test)
        test1 = self.trans_2(test)
        #torchvision.utils.save_image(test1, full_address_test + str(idx) + '_2_r.png')
        torchvision.utils.save_image(test1, './rotation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png')

        return ref1, test1, gt

class geo_dilate(Dataset):
    # without norm
    def __init__(self, jnd_info, root_dir, test=True):
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        self.test = test
        if test == False:
            self.trans_1 = transforms.Compose([
                transforms.ToTensor(),

            ])
            self.trans_2 = transforms.Compose([
                transforms.ToTensor(),

            ])
        else:
            self.trans_1 = transforms.Compose([
                transforms.CenterCrop(896),
                transforms.ToTensor(),
                ])
            self.trans_2 = transforms.Compose([
                transforms.CenterCrop(940),
                transforms.Resize(1000),
                transforms.CenterCrop(896),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        print(idx)
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        ref = Image.open(full_address)
        ref1 = self.trans_1(ref)
        #torchvision.utils.save_image(ref1, full_address + str(idx) + '_1_d.png')
        torchvision.utils.save_image(ref1, './dilation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_1.png')

        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test)
        test1 = self.trans_2(test)
        #torchvision.utils.save_image(test1, full_address_test + str(idx) + '_2_d.png')
        torchvision.utils.save_image(test1, './dilation/' + str(self.ref_name[idx]) + '_' + str(idx) + '_2.png')

        return ref1, test1, gt


class CD_npy(Dataset):
    # without norm
    def __init__(self, jnd_info, root_dir, test=False):
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        self.test = test
        self.trans1 = transforms.ToTensor()
        random.seed(20)
        torch.random.manual_seed(20)
    def __len__(self):
        return len(self.gt)
    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        ref = np.load(os.path.join(self.root_dir, str(self.ref_name[idx])), allow_pickle=True)
        ref = self.trans1(ref/255)
        # ref = self.trans1(ref )
        test = np.load(os.path.join(self.root_dir, str(self.test_name[idx])), allow_pickle=True)
        test = self.trans1(test/255)
        # test = self.trans1(test)
        return ref, test, gt

class CD_resnext(Dataset):
    def __init__(self, jnd_info, root_dir, test=False):
        # Initialization of file names and dis-similarity measure acquired from MCL-JCI dataset.
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]
        random.seed(20)
        torch.random.manual_seed(20)
        if test == False:
            self.transform = transforms.Compose([
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                transforms.RandomCrop(512),
                # transforms.Resize(256),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(768),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        ref = self.transform(ref)

        full_address = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        test = self.transform(test)
        return ref, test, gt
class test_1024(Dataset):
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
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.ToTensor(),
                ])
    def __len__(self):
        return len(self.gt)
    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # ref1 = self.trans_org(ref.rotate(90))
        ref1 = self.trans_org(ref)
        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # test1 = self.trans_org(test.rotate(90))
        test1 = self.trans_org(test)
        return ref1, test1, gt
class test_768(Dataset):
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
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                transforms.Resize(1024),
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                transforms.Resize(768),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # ref1 = self.trans_org(ref.rotate(90))
        ref1 = self.trans_org(ref)
        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # test1 = self.trans_org(test.rotate(90))
        test1 = self.trans_org(test)
        return ref1, test1, gt
class test_512(Dataset):
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
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                # transforms.RandomCrop(512),
                transforms.Resize(1024),
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                # transforms.RandomHorizontalFlip(p=1),
                # transforms.RandomVerticalFlip(p=1),
                transforms.Resize(512),
                # transforms.Resize(768),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # ref1 = self.trans_org(ref.rotate(90))
        ref1 = self.trans_org(ref)
        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # test1 = self.trans_org(test.rotate(90))
        test1 = self.trans_org(test)
        return ref1, test1, gt
class test_256(Dataset):
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
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                # transforms.RandomCrop(512),
                transforms.Resize(1024),
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                # transforms.RandomHorizontalFlip(p=1),
                # transforms.RandomVerticalFlip(p=1),
                transforms.Resize(256),
                # transforms.Resize(768),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # ref1 = self.trans_org(ref.rotate(90))
        ref1 = self.trans_org(ref)
        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # test1 = self.trans_org(test.rotate(90))
        test1 = self.trans_org(test)
        return ref1, test1, gt
class test_128(Dataset):
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
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5), 
                transforms.RandomInvert(p=0.5),
                # transforms.RandomCrop(512),
                transforms.crop(128),
                transforms.ToTensor(),

            ])
        else:
            self.trans_org = transforms.Compose([
                # transforms.Resize(64),
                # transforms.Resize(768),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        full_address = os.path.join(self.root_dir, str(self.ref_name[idx]))
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        ref = Image.open(full_address).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        ref1 = self.trans_org(ref)

        full_address_test = os.path.join(self.root_dir, str(self.test_name[idx]))
        test = Image.open(full_address_test).convert("RGB")
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        test1 = self.trans_org(test)
        return ref1, test1, gt
