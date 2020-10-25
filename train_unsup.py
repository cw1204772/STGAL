import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
cudnn.benchmark=True

import sys
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np
from collections import OrderedDict

from utils import *
from loss import TripletLoss
from logger import Logger
import models
from cmc import Self_Cmc, Vanilla_Cmc
import inference

def vanilla_validation(q_dataloader, q_txt, g_dataloader, g_txt, base_net, rank_size):
    q_features = inference.inference(q_dataloader, base_net)
    g_features = inference.inference(g_dataloader, base_net)
    CMC, mAP = Vanilla_Cmc(q_features, q_txt, g_features, g_txt, rank_size=rank_size)
    return CMC[0], mAP

def self_validation(dataloader, txt, base_net, rank_size):
    features = inference.inference(dataloader, base_net)
    CMC, mAP = Self_Cmc(features, txt, rank_size=rank_size)
    return CMC[0], mAP

def train_loop(args, tgt_dataloader, base_net):

    optimizer_base = optim.SGD(base_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_base, args.lr_step_size, 0.1)
    criterion_triplet = TripletLoss(margin=args.margin, batch_hard=args.batch_hard)
    loss_logger = Logger(args.save_model_dir, 'loss.csv')
    acc_logger = Logger(args.save_model_dir, 'acc.csv')
   
    best_tgt_r1 = 0.0 
    for it in range(args.n_iters):
        # Validation
        if it % args.test_every_n_iter == 0:
            tgt_train_r1, tgt_train_mAP = self_validation(tgt_dataloader['train'], args.tgt_train, base_net, args.rank_size)
            print('(iter %d)[Train] Target r-1 accuracy: %.3f, mAP: %.3f' % (it, tgt_train_r1, tgt_train_mAP))
            if args.tgt_val is not None:
                tgt_r1, tgt_mAP = self_validation(tgt_dataloader['val'], args.tgt_val, base_net, args.rank_size)
            else:
                tgt_r1, tgt_mAP = vanilla_validation(tgt_dataloader['query'], args.tgt_query,
                                                     tgt_dataloader['gallery'], args.tgt_gallery, base_net, args.rank_size)
            print('(iter %d)[Val] Target r-1 accuracy: %.3f, mAP: %.3f' % (it, tgt_r1, tgt_mAP))
            log = OrderedDict([('tgt train r-1', tgt_train_r1),
                              ('tgt train mAP', tgt_train_mAP),
                              ('tgt val r-1', tgt_r1),
                              ('tgt val mAP', tgt_mAP)])
            acc_logger.logg(it, log)
            loss_logger.write_log()
            acc_logger.write_log()
            
            if tgt_r1 > best_tgt_r1:
                torch.save(base_net.state_dict(),os.path.join(args.save_model_dir,'model_best_base.ckpt'))
                best_tgt_r1 = tgt_r1
        
        log = {}
       
        data = next(tgt_dataloader['train_triplet'])
        b_img = data['img'].cuda()
        pos_mask = data['pos_mask'].cuda()
        neg_mask = data['neg_mask'].cuda()
        #forward
        pred_feat = base_net(b_img)
        b_loss = criterion_triplet(pred_feat, pos_mask=pos_mask, neg_mask=neg_mask, mode='mask')
        loss = b_loss.mean()
        # backward
        base_net.zero_grad()
        loss.backward()
        optimizer_base.step()

        log['loss_tgt_triplet'] = b_loss.data.mean().item()
        log['loss_tgt_triplet_max'] = b_loss.data.max().item()
        log['loss_tgt_triplet_min'] = b_loss.data.min().item()
        
        scheduler.step()
        loss_logger.logg(it, log)
        print('(iter %d)[Train] Target triplet loss = %.3e' %
              (it, log['loss_tgt_triplet']))
        


def parse_args():
    parser = argparse.ArgumentParser(description='Train Re-ID net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tgt_pkl', help='pkl for target dataset (train)')
    parser.add_argument('--tgt_train', help='txt for target dataset (train)')
    parser.add_argument('--tgt_val', help='txt for target dataset (val)')
    parser.add_argument('--tgt_query', help='txt for target dataset (val)')
    parser.add_argument('--tgt_gallery', help='txt for target dataset (val)')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--lr_step_size',type=int,default=5000,help='step size for stepLR')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size number')
    parser.add_argument('--n_iters',type=int,default=20,help='number of training iterations')
    parser.add_argument('--load_ckpt',default=None,help='path to load ckpt')
    parser.add_argument('--save_model_dir',default=None,help='path to save model')
    parser.add_argument('--n_layer',type=int,default=18,help='number of Resnet layers')
    parser.add_argument('--margin',type=str,default='0',help='margin of triplet loss ("soft" or float)')
    parser.add_argument('--class_per_batch',type=int,default=32,help='# of class per batch for triplet training')
    parser.add_argument('--image_per_class',type=int,default=4,help='# of images per class for triplet training')
    parser.add_argument('--batch_hard',action='store_true',help='whether to use batch_hard for triplet loss')
    parser.add_argument('--display_every_n_iter',type=int,default=1,help='display every n iterations')
    parser.add_argument('--test_every_n_iter',type=int,default=200,help='test every n iterations')
    parser.add_argument('--save_every_n_iter',type=int,default=1000,help='save model every n iterations')
    parser.add_argument('--pretrain_model',type=str,default=None,help='load pretrained model')
    parser.add_argument('--sample_mode',type=str,choices=['fix','unfix'],default='unfix',help='choose sampling model for CTM dataloader')
    parser.add_argument('--drop_ctm', type=bool, default=False, help='do not train CTM')
    parser.add_argument('--rank_size', type=int, default=100, help='rank size for CMC and mAP')
    parser.add_argument('--fc_dim', type=int, default=0, help='adding a fc of # dim to resnet50')
    args = parser.parse_args()
    args.margin = args.margin if args.margin=='soft' else float(args.margin)
    if args.class_per_batch * args.image_per_class != args.batch_size:
        sys.exit('batch_size need to equal class_in_batch*image_per_class_in_batch')
    if args.save_model_dir != None:
        os.system('mkdir -p %s' % os.path.join(args.save_model_dir))
    return args

if __name__ == '__main__':
    # Parse arg
    args = parse_args()

    # Set transformation
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Get target dataLoader 
    tgt_dataloader = {}
    tgt_dataloader['train_triplet'] = Get_unsupervised_triplet_DataLoader(args.tgt_pkl, train_transform, args.batch_size,
                                                                          args.sample_mode, args.image_per_class)
    tgt_dataloader['train'] = Get_normal_DataLoader(args.tgt_train, test_transform, batch_size=64)
    if args.tgt_val is not None:
        tgt_dataloader['val'] = Get_normal_DataLoader(args.tgt_val, test_transform, batch_size=64)
    else:
        tgt_dataloader['query'] = Get_normal_DataLoader(args.tgt_query, test_transform, batch_size=64)
        tgt_dataloader['gallery'] = Get_normal_DataLoader(args.tgt_gallery, test_transform, batch_size=64)

    # Get Model
    #base_net = models.new_FeatureResNet(n_layers=args.n_layer, pretrained=True, fc_dim=args.fc_dim)
    base_net = models.FeatureResNet(n_layers=args.n_layer, pretrained=True)
    if args.pretrain_model is not None:
        pretrain_model = torch.load(args.pretrain_model)
        for key in ['fc.weight','fc.bias']:
            if key in pretrain_model:
                del pretrain_model[key]
        base_net.load_state_dict(pretrain_model)
        print('pretrain model loaded.')
    base_net = base_net.cuda()

    # Train
    train_loop(args, tgt_dataloader, base_net)

