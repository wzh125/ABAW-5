import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from conf import set_logger,set_outdir,set_env
from model.ANFL import MAE_Graph
from utils import *
from dataset.dataset import video_Aff_Wild2_train, video_Aff_Wild2_val

import argparse

def par():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Datasets
    parser.add_argument('--task', default='AU')
    parser.add_argument('--dataset', default="Aff", type=str, help="experiment dataset BP4D / DISFA")
    parser.add_argument('--dataset_path', default='/raid/wangzihan/5th_ABAW')
    # Param
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR',help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num_workers', default=10, type=int, metavar='N',help='number of data loading workers (default: 4)')

    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W',help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float)
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('--length',default=16, type=int, help='frame number of each clip')
    parser.add_argument('--au_num', default=12, type=int, help='number of au')
    parser.add_argument('--neighbor_num',default=4,type=int, help='number of neighbor of each au')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # Network and Loss
    parser.add_argument('--arc', default='mae', type=str, help="backbone architecture mae / swin_transformer")
    # Device and Seed
    parser.add_argument('--gpu_ids', type=str, default='2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

    # Experiment
    parser.add_argument('--exp-name', default='', type=str,
                        help="experiment name for saving checkpoints")
    parser.add_argument('--resume', default='', type=str, metavar='path',
                        help='path to latest checkpoint (default: none)')

    return parser.parse_args()

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'Aff':
        trainset = video_Aff_Wild2_train(task='AU',root_path=conf.dataset_path, length=conf.length, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)

        valset = video_Aff_Wild2_val(task='AU',root_path=conf.dataset_path, length=conf.length, transform=image_test(crop_size=conf.crop_size),crop_size=conf.crop_size)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    else:
        raise ValueError('datasey error')
    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets, = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.view(-1,12))
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), outputs.size(0))
    return losses.avg

# Val
def val(net,val_loader,criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets.view(-1,12))
            losses.update(loss.data.item(), outputs.size(0))
            update_list = statistics(outputs, targets.view(-1,12).detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):

    dataset_info = Aff_Wild2_infolist
    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'annotation', 'AU_Detection_Challenge',
                                                            'Aff_Wild2_AU_train_weight.txt')))

    net = MAE_Graph(task=conf.task, num_classes=conf.au_num, backbone=conf.arc, neighbor_num=conf.neighbor_num)
    #net = swin(task=conf.task)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    #criterion = nn.BCELoss()
    criterion = WeightedAsymmetricLoss()#weight=train_weight)
    optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        infostr = {'Epoch:  {}   train_loss: {:.5f}  '.format(epoch + 1, train_loss)}
        logging.info(infostr)

        # val and save checkpoints
        if (epoch+1) % 100 == 0:
            val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)
            infostr = {'epoch: {} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'.format(epoch+1, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}
            logging.info(infostr)
            infostr = {'F1-score-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_f1_score)
            logging.info(infostr)
            infostr = {'Acc-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_acc)
            logging.info(infostr)

            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf.outdir, 'epoch' + str(epoch + 1) + '_model.pth'))

if __name__ == '__main__':
    conf = par()
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_ids

    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
