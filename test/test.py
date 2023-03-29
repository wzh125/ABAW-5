import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model.ANFL import MAE_Graph
from utils import *
from dataset import video_Aff_Wild2_test

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
    parser.add_argument('--resume', default='/raid/wangzihan/video_GraphAU/results/mae/bs_8_seed_0_lr_0.0001/epoch200_model.pth', type=str, metavar='path',
                        help='path to latest checkpoint (default: none)')

    return parser.parse_args()

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'Aff':

        testset = video_Aff_Wild2_test(task='AU',root_path=conf.dataset_path, length=conf.length, transform=image_test(crop_size=conf.crop_size),crop_size=conf.crop_size)
        test_loader = DataLoader(testset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    else:
        raise ValueError('dataset error')
    return test_loader, len(testset)

# Val
def val(net,val_loader):

    net.eval()
    result_pre = []
    for batch_idx, (inputs, num) in enumerate(tqdm(val_loader)):
        b = inputs.size(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = net(inputs).view(b, 16, 12)
            outputs = (outputs >= 0.5)
            outputs = np.asarray(outputs.cpu(), dtype=int)

            for batch in range(b):
                for frame in range(num[batch]):
                    result_pre.append(outputs[batch][frame])
    return result_pre

def main(conf):

    test_loader, test_data_num = get_dataloader(conf)

    net = MAE_Graph(task=conf.task, num_classes=conf.au_num, backbone=conf.arc, neighbor_num=conf.neighbor_num)

    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_mae_graph(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    result_pre = val(net, test_loader)

    with open(conf.dataset_path+'/test/clear_result_4.txt','w') as f:
        for i in range(len(result_pre)):
            res = ','.join([str(k) for k in result_pre[i]])
            f.writelines(res)
            f.write('\n')

if __name__ == '__main__':
    conf = par()
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_ids
    main(conf)
