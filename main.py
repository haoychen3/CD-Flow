import torch
from trainnet import trainNet
import pandas as pd
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--scheduler_step", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--batch_size_train", type=int, default=4)
    parser.add_argument("--batch_size_test", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--training_datadir", type=str, default='')
    parser.add_argument("--colorspace", type=str, default='rgb')
    parser.add_argument("--trainpath1", type=str, default='trainnet.py')
    parser.add_argument("--trainpath2", type=str, default='main.py')
    parser.add_argument("--trainpath3", type=str, default='model.py')
    parser.add_argument("--trainpath4", type=str, default='DataLoader.py')
    parser.add_argument("--work_path", type=str, default='work_dir')

    parser.add_argument("--datapath", type=str, default='data')
    parser.add_argument("--trainset", type=str, default='train.csv')
    parser.add_argument("--valset", type=str, default='val.csv')
    parser.add_argument("--testset", type=str, default='test.csv')
    parser.add_argument("--test_aligned_path", type=str, default=None)
    parser.add_argument("--test_notaligned_path", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_config()
    path = config.datapath
    modelprediction = pd.DataFrame(columns=['no'])
    modelprediction_aligned = pd.DataFrame(columns=['no'])
    modelprediction_notaligned = pd.DataFrame(columns=['no'])
    work_path = config.work_path
    trainpath = config.trainset
    valpath = config.valset
    testpath = config.testset
    performance = pd.DataFrame(columns=['stress', 'plcc', 'srcc', 'stress_aligned', 'plcc_aligned', 'srcc_aligned', 'stress_notaligned', 'plcc_notaligned', 'srcc_notaligned'])
    torch.cuda.empty_cache()
    i = 0
    config.datapath = path+'/{}.csv'.format(i+1)
    config.work_path = work_path+'/{}'.format(i+1)
    config.trainset = path+'/{}/'.format(i+1)+trainpath
    config.valset = path+'/{}/'.format(i+1)+valpath
    config.testset = path+'/{}/'.format(i+1)+testpath
    config.test_aligned_path = path+'/{}/test_aligned.csv'.format(i+1)
    config.test_notaligned_path = path+'/{}/test_notaligned.csv'.format(i+1)
    dist1, y_true1, stress1, cc_v1, srocc_v1, dist2, y_true2, stress2, cc_v2, srocc_v2,\
    dist3, y_true3, stress3, cc_v3, srocc_v3 = trainNet(config, i)

    performance.loc['{}'.format(i), 'stress'] = stress1
    performance.loc['{}'.format(i), 'plcc'] = cc_v1
    performance.loc['{}'.format(i), 'srcc'] = srocc_v1
    performance.loc['{}'.format(i), 'stress_aligned'] = stress2
    performance.loc['{}'.format(i), 'plcc_aligned'] = cc_v2
    performance.loc['{}'.format(i), 'srcc_aligned'] = srocc_v2
    performance.loc['{}'.format(i), 'stress_notaligned'] = stress3
    performance.loc['{}'.format(i), 'plcc_notaligned'] = cc_v3
    performance.loc['{}'.format(i), 'srcc_notaligned'] = srocc_v3
    performance.to_csv(config.work_path + '/modelperformance.csv', index=None)
