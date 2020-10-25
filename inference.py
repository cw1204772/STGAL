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

import utils
import models

def inference(dataloader, base_net):
    """Inference dataloader"""
    base_net.eval()
    with torch.no_grad():
        features = [] 
        for samples in tqdm(dataloader, ncols=100):
            b_img = samples['img'].cuda()
            pred_feat = base_net(b_img)
            features.append(pred_feat)
        features = torch.cat(features, dim=0).cpu().numpy()
    base_net.train()
    return features

def inference_db(db_txt, model):
    """Inference images in database txt"""
    # Set transformation
    test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Get Dataset & DataLoader 
    dataloader = utils.Get_normal_DataLoader(db_txt, test_transform, batch_size=64)

    # Get Model
    base_net = models.FeatureResNet(n_layers=50)
    pretrain_model = torch.load(model)
    for key in ['fc.weight','fc.bias']:
        if key in pretrain_model:
            del pretrain_model[key]
    base_net.load_state_dict(pretrain_model)
    print('pretrain model loaded.')
    base_net = base_net.cuda()

    # Inference
    features = inference(dataloader, base_net)
    return features

