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
import torch.autograd as autograd
from function import setup_seed, copy_codes
from math import log


def trainNet(config, times):
    resume_path = config.resume_path
    learning_rate = config.learning_rate
    scheduler_step = config.scheduler_step
    scheduler_gamma = config.scheduler_gamma
    batch_size_train = config.batch_size_train
    batch_size_test = config.batch_size_test
    n_epochs = config.n_epochs
    training_datadir = config.training_datadir
    colorspace = config.colorspace
    trainpath1 = config.trainpath1
    trainpath2 = config.trainpath2
    trainpath3 = config.trainpath3
    trainpath4 = config.trainpath4
    workspace = config.work_path
    device = torch.device("cuda")
    # set random seed
    setup_seed(config.seed)
    if not os.path.exists(workspace):
        os.mkdir(workspace)
    if not os.path.exists(os.path.join(workspace, 'codes')):
        os.mkdir(os.path.join(workspace, 'codes'))
    if not os.path.exists(os.path.join(workspace, 'checkpoint')):
        os.mkdir(os.path.join(workspace, 'checkpoint'))
    if not os.path.exists(os.path.join(workspace, 'checkpoint_best')):
        os.mkdir(os.path.join(workspace, 'checkpoint_best'))
    copy_codes(trainpath1=trainpath1, trainpath2=trainpath2, trainpath3=trainpath3, trainpath4=trainpath4,
               path1=os.path.join(workspace, 'codes/trainNet.py'), path2=os.path.join(workspace, 'codes/main.py'),
               path3=os.path.join(workspace, 'codes/net.py'), path4=os.path.join(workspace, 'codes/DataLoader.py'))

    print("============ HYPERPARAMETERS ==========")
    print("batch_size_train and test=", batch_size_train, batch_size_test)
    print("epochs=", n_epochs)
    print('learning rate=', learning_rate)
    print('scheduler_step=', scheduler_step)
    print('scheduler_gamma=', scheduler_gamma)
    print('training dir=', training_datadir)
    print('colorspace=', colorspace)
    print(config.trainset)
    print(config.valset)
    print(config.testset)
    print(config.test_aligned_path)
    print(config.test_notaligned_path)
    train_pairs = np.genfromtxt(open(config.trainset, encoding='UTF-8-sig'), delimiter=',', dtype=str)
    val_pairs = np.genfromtxt(open(config.valset, encoding='UTF-8-sig'), delimiter=',', dtype=str)
    test_pairs = np.genfromtxt(open(config.testset, encoding='UTF-8-sig'), delimiter=',', dtype=str)

    test_aligned_pairs = np.genfromtxt(open(config.test_aligned_path), delimiter=',', dtype=str)
    test_notaligned_pairs = np.genfromtxt(open(config.test_notaligned_path), delimiter=',', dtype=str)

    data_train = CD_128(train_pairs[:], root_dir=training_datadir, test=False)
    data_val = CD_128(val_pairs[:], root_dir=training_datadir, test=True)
    data_test = CD_128(test_pairs[:], root_dir=training_datadir, test=True)
    test_aligned = CD_128(test_aligned_pairs[:], root_dir=training_datadir, test=True)
    test_notaligned = CD_128(test_notaligned_pairs[:], root_dir=training_datadir, test=True)

    net = CDFlow().to(device)
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    loss, optimizer, scheduler = createLossAndOptimizer(net, learning_rate, scheduler_step, scheduler_gamma)

    data_train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True,
                                   pin_memory=True, num_workers=4)
    data_val_loader = DataLoader(data_val, batch_size=batch_size_test, shuffle=True, pin_memory=True,
                                 num_workers=4)
    data_test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False,
                                  pin_memory=True, num_workers=4)
    data_test_aligned_loader = DataLoader(test_aligned, batch_size=batch_size_test, shuffle=False,
                                          pin_memory=True, num_workers=4)
    data_test_notaligned_loader = DataLoader(test_notaligned, batch_size=batch_size_test,
                                             shuffle=False, pin_memory=True, num_workers=4)

    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('continue to train: shuffle{} epoch{} '.format(times + 1, start_epoch))
    else:
        start_epoch = 0

    training_start_time = time.time()
    rows, columns = train_pairs.shape
    n_batches = rows // batch_size_train
    valsrcc = 0
    ema = EMA(net, 0.999)
    ema.register()
    autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, n_epochs):
        # initiate parameters for statistic recordings.
        dist = []
        y_true = []
        running_loss = 0.0
        total_train_loss = 0
        start_time = time.time()
        print_every = 20
        train_counter = 0
        net.train()
        print("---------------------train mode-------epoch{}--------------------------".format(epoch))
        for i, data in enumerate(data_train_loader, 0):
            train_counter = train_counter + 1
            x, y, gts = data
            y_val = gts.numpy()
            x, y, gts = \
                Variable(x).to(device), \
                Variable(y).to(device), \
                Variable(gts).to(device)
            optimizer.zero_grad()

            score, score65432, score6543, score654, score65, score6, log_p_x, logdet_x, log_p_y, logdet_y = net(x, y)

            logdet_x = logdet_x.mean()
            logdet_y = logdet_y.mean()

            loss_x, log_p_x, log_det_x = calc_loss(log_p_x, logdet_x, 768, 2.0 ** 5)
            loss_y, log_p_y, log_det_y = calc_loss(log_p_y, logdet_y, 768, 2.0 ** 5)

            score_loss = 10 * loss(score, gts) + loss(score65432, gts) + loss(score6543, gts) + loss(score654, gts) + loss(score65, gts) + loss(score6, gts)

            loss_size = score_loss + loss_x + loss_y
            loss_size.backward()
            optimizer.step()
            ema.update()

            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            pred = (torch.squeeze(score)).cpu().detach().numpy().tolist()
            if isinstance(pred, list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)

            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.6f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))

                running_loss = 0.0
                start_time = time.time()

        torch.save(
            {"state_dict": net.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'times': times}, \
            os.path.join(workspace, 'checkpoint', 'ModelParams_checkpoint.pt'))

        # Calculate correlation coefficients between the predicted values and ground truth values on training set.
        dist = np.array(dist).squeeze()
        y_true = np.array(y_true).squeeze()
        _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist, y_true)
        print("Training set: PCC{:.4}, SROCC{:.4}, KROCC{:.4}, RMSE{:.4}".format(cc_v, srocc_v, krocc_v, rmse_v))
        # validation
        # EMA
        ema.apply_shadow()
        # EMA
        net.eval()
        print("----------------------------validation mode---------------------------------")
        srocc_v, total_val_loss, val_counter, cc_v, krocc_v, rmse_v, stress, dist, y_true, score_val = test(
            data_val_loader, net, loss)
        # srocc_a, total_val_loss_a, val_counter_a, cc_a, krocc_a, rmse_a, stress_a, dist_a, y_true_a, score_a = test(
        #     data_test_aligned_loader, net, loss)
        # srocc_na, total_val_loss_na, val_counter_na, cc_na, krocc_na, rmse_na, stress_na, dist_na, y_true_na, score_na = test(
        #     data_test_notaligned_loader, net, loss)

        if srocc_v > valsrcc:
            valsrcc = srocc_v
            torch.save({"state_dict": net.state_dict()},
                       os.path.join(workspace, 'checkpoint_best', 'ModelParams_Best_val.pt'))
            print('update  best model...')
        print("VALIDATION:  PCC{:.4}, SROCC{:.4}, STRESS{:.4}, RMSE{:.4}".format(cc_v, srocc_v, stress, rmse_v))
        print("loss = {:.6}".format(total_val_loss / val_counter))
        # EMA
        ema.restore()
        # EMA
        scheduler.step()

    print('#############################################################################')
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    pt = os.path.join(workspace, 'checkpoint_best', 'ModelParams_Best_val.pt')
    checkpoint = torch.load(pt)
    net = CDFlow().to(device)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    srocc_v1, total_val_loss, val_counter, cc_v1, krocc_v, rmse_v, stress1, dist1, y_true1, score_val = test(
        data_test_loader, net, loss)
    print('best performance: plcc{} srcc{}'.format(cc_v1, srocc_v1))
    srocc_v2, total_val_loss, val_counter, cc_v2, krocc_v, rmse_v, stress2, dist2, y_true2, score_val = test(
        data_test_aligned_loader, net, loss)
    print('best performance in Pixel-wise aligned: plcc{} srcc{}'.format(cc_v2, srocc_v2))
    srocc_v3, total_val_loss, val_counter, cc_v3, krocc_v, rmse_v, stress3, dist3, y_true3, score_val = test(
        data_test_notaligned_loader, net, loss)
    print('best performance in non-Pixel-wise aligned: plcc{} srcc{}'.format(cc_v3, srocc_v3))
    return dist1, y_true1, stress1, cc_v1, srocc_v1, dist2, y_true2, stress2, cc_v2, srocc_v2, dist3, y_true3, stress3, cc_v3, srocc_v3


def test(data_val_loader, net, loss):
    total_val_loss = 0
    val_counter = 0
    score_val = 0
    dist = []
    y_true = []
    device = torch.device("cuda")
    for i, data in enumerate(data_val_loader, 0):
        with torch.no_grad():
            x, y, gts = data
            y_val = gts.numpy()
            x, y, gts = \
                Variable(x).to(device), \
                Variable(y).to(device), \
                Variable(gts).to(device)

            score, _, _, _, _, _, _, _, _, _ = net(x, y)

            score_loss = loss(score, gts)
            loss_size = score_loss
            total_val_loss += loss_size.cpu().numpy()
            score_val = score_val + score_loss.item()
            val_counter += 1
            pred = (torch.squeeze(score)).cpu().detach().numpy().tolist()
            if isinstance(pred, list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)
    # Calculate correlation coefficients between the predicted values and ground truth values on validation set.
    dist_np = np.array(dist).squeeze()
    y_true_np = np.array(y_true).squeeze()
    stress = compute_stress(dist_np, y_true_np)
    _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist_np, y_true_np)
    return srocc_v, total_val_loss, val_counter, cc_v, krocc_v, rmse_v, stress, dist, y_true, score_val


def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (-loss / (log(2) * n_pixel)).mean(), (log_p / (log(2) * n_pixel)).mean(), (
                logdet / (log(2) * n_pixel)).mean()
