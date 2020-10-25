import os
import glob
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import Compose,Resize,ToTensor,Normalize
from PIL import Image,ImageEnhance
import time as time
import models
import numpy as np
import torchvision.utils as vutils
import pickle
import pandas as pd
import random

##### Parse & Preprocess #####
def parse_db(db_txt, keys):
    """Read data from database txt"""
    df = pd.read_csv(db_txt, sep=' ')
    db = {k:df.loc[:, k].values for k in keys}
    return db

def process_labels(labels):
    unique_id = np.unique(labels)
    print('total id count:', len(unique_id))
    id_dict = {ID:i for i, ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels

def hash_track_id(track_id):
    return 1e13+track_id[0]*1e+12 + track_id[1]*1e+6 + track_id[2]

##### Dataset #####
class Unsupervised_Triplet_Image_Dataset(object):
    def __init__(self, pkl_name, transform):
        with open(pkl_name, 'rb') as f:
            data = pickle.load(f)
        self.samples = data['afl_samples']
        self.imgs = data['track_dict']
        self.transform = transform
    def getitem(self, track_id, img_idx):
        return self.transform(Image.open(self.imgs[track_id][img_idx]))
    def __len__(self):
        return sum([len(track) for track in self.imgs])

class Image_Dataset(Dataset):
    def __init__(self, db_txt, transform, image_per_class=0, filter_labels=False):
        with open(db_txt, 'r') as f:
            df = pd.read_csv(f, sep=' ')
            df = df.groupby('label').filter(lambda x: len(x) > image_per_class)
            if filter_labels:
                df = df.loc[df.loc[:, 'label']>0]
            
        self.imgs = df.loc[:, 'img'].values
        self.labels = process_labels(df.loc[:, 'label'].values) # Process labels
        
        self.transform = transform 

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = int(self.labels[idx])
        return {'img':self.transform(img), 'class':label}

    def __len__(self):
        return len(self.labels)

class PCTD_Dataset(Dataset):
    def __init__(self, db_txt, transform, image_per_class=0, filter_labels=False):
        with open(db_txt, 'r') as f:
            df = pd.read_csv(f, sep=' ')
            df = df.groupby('label').filter(lambda x: len(x) > image_per_class)
            if filter_labels:
                df = df.loc[df.loc[:, 'label']>0]
            
        imgs = df.loc[:, 'img'].values
        cam = df.loc[:, 'cam'].values
        label = df.loc[:, 'label'].values
        track_id = df.loc[:, 'track_id'].values
        self.img_dict = {}
        self.label_dict = {}

        for cam_id in np.unique(cam):
            idx = cam == cam_id
            uuid = process_labels(label[idx]*1000 + track_id[idx])
            self.img_dict[cam_id] = imgs[idx]
            self.label_dict[cam_id] = uuid
            assert len(self.img_dict[cam_id] == self.label_dict[cam_id])
        
        self.transform = transform 

    def getitem(self, cam_id, idx):
        img = Image.open(self.img_dict[cam_id][idx])
        label = int(self.label_dict[cam_id][idx])
        return {'img':self.transform(img), 'class':label}

    def get_n_ids_list(self):
        n_ids_list = []
        for cam_id in sorted(self.label_dict.keys()):
            n_ids_list.append(len(np.unique(self.label_dict[cam_id])))
        return n_ids_list

##### Dataloader #####
class Unsupervised_Triplet_Image_DataLoader(object):
    def __init__(self, dataset, max_batch_size, mode='unfix', image_per_class=None):
        self.dataset= dataset
        self.samples = dataset.samples
        self.imgs = dataset.imgs
        self.max_batch_size = max_batch_size
        self.image_per_class = image_per_class
        self.mode = mode
    def __next__(self):
        img_list = []
        sample_list = []
        while len(img_list) < self.max_batch_size:
            sample_idx = random.choice(range(len(self.samples)))
            tracks = self.samples[sample_idx]
            track_idx = np.random.permutation(len(tracks))
            for i in track_idx:
                track_id = tracks[i]
                if len(img_list) >= self.max_batch_size:
                    break
                if self.mode == 'unfix':
                    for img_idx in range(len(self.imgs[track_id])):
                        img_list.append((track_id, img_idx))
                        sample_list.append(sample_idx)
                elif self.mode == 'fix':
                    if len(self.imgs[track_id]) == 0:
                        raise RuntimeError('a track should have at least 1 image')
                    for i in range(self.image_per_class):
                        if i < len(self.imgs[track_id]):
                            img_list.append((track_id, i))
                        else:
                            img_list.append((track_id, len(self.imgs[track_id])-1))
                        sample_list.append(sample_idx)
                else:
                    raise NotImplementedError('mode need to be either "fix" or "unfix"')
        track_list = [hash_track_id(track_id) for track_id, _ in img_list]

        # Shuffle
        rng = np.random.permutation(len(img_list))
        #rng = np.random.choice(len(img_list), size=self.max_batch_size, replace=False)
        img_list = np.array(img_list)[rng]
        track_list = np.array(track_list)[rng]
        sample_list = np.array(sample_list)[rng]

        # Cut to fit batch
        img_list = img_list[:self.max_batch_size]
        sample_list = sample_list[:self.max_batch_size]
        track_list = track_list[:self.max_batch_size]

        # Generate mask
        imgs = [self.dataset.getitem(track_idx, img_idx) for track_idx, img_idx in img_list]
        sample_list = sample_list.reshape(-1, 1)
        sample_list = (sample_list == sample_list.T)
        track_list = track_list.reshape(-1, 1)
        track_list = (track_list == track_list.T)
        pos_mask = (sample_list & track_list) ^ np.eye(len(img_list), dtype=bool)
        neg_mask = sample_list & (~track_list)
        return {'img':torch.stack(imgs, dim=0), 
                'pos_mask': torch.from_numpy(pos_mask.astype(np.uint8)),
                'neg_mask': torch.from_numpy(neg_mask.astype(np.uint8))}
    def __iter__(self):
        return self

class CDM_Triplet_Image_DataLoader(object):
    def __init__(self, src_dataset, tgt_dataset, batch_size, image_per_class=None, drop_afl=False):
        # source dataset
        self.src_dataset= src_dataset
        self.src_img_list = {}
        for c in np.unique(src_dataset.labels):
            idxs = np.nonzero(src_dataset.labels == c)[0]
            if len(idxs) >= image_per_class:
                self.src_img_list[c] = idxs
        self.src_unique_labels = np.array(list(self.src_img_list.keys()))

        # target dataset
        self.tgt_dataset= tgt_dataset
        self.tgt_samples = tgt_dataset.samples
        self.tgt_imgs = tgt_dataset.imgs

        self.drop_afl = drop_afl
        self.batch_size = batch_size
        self.src_batch_size = ((batch_size//2)//image_per_class) * image_per_class
        self.tgt_batch_size = batch_size - self.src_batch_size
        self.image_per_class = image_per_class

    def __next__(self):
        label_list = np.zeros((self.batch_size)) # record class for source, track_id for target

        # source dataset
        classes = np.random.choice(self.src_unique_labels, self.src_batch_size//self.image_per_class)
        src_img_list = []
        src_label_list = []
        for i, c in enumerate(classes):
            src_img_list += np.random.choice(self.src_img_list[c], self.image_per_class).tolist()
            src_label_list += ([c] * self.image_per_class)
        assert len(src_img_list) == self.src_batch_size
        src_img_list = np.array(src_img_list)
        label_list[:self.src_batch_size] = np.array(src_label_list)

        # target dataset
        tgt_img_list = []
        tgt_sample_list = []
        while len(tgt_img_list) < self.tgt_batch_size:
            sample_idx = random.choice(range(len(self.tgt_samples)))
            for track_id in self.tgt_samples[sample_idx]:
                if len(self.tgt_imgs[track_id]) == 0:
                    raise RuntimeError('a track should have at least 1 image')
                for i in range(self.image_per_class):
                    img_idx = min(i, len(self.tgt_imgs[track_id])-1)
                    tgt_img_list.append((track_id, img_idx))
                    tgt_sample_list.append(sample_idx)
        tgt_label_list = [hash_track_id(track_id) for track_id, _ in tgt_img_list]
        tgt_img_list = np.array(tgt_img_list)[:self.tgt_batch_size]
        tgt_sample_list = np.array(tgt_sample_list)[:self.tgt_batch_size]
        label_list[self.src_batch_size:] = np.array(tgt_label_list)[:self.tgt_batch_size]

        # Read img
        imgs = []#torch.FloatTensor(self.batch_size, 3, 224, 224)
        for i, idx in enumerate(src_img_list):
            imgs.append(self.src_dataset[int(idx)]['img'])
        for i, (track_idx, img_idx) in enumerate(tgt_img_list):
            imgs.append(self.tgt_dataset.getitem(track_idx, img_idx))
        imgs = torch.stack(imgs, dim=0)

        # Generate mask
        sample_mask = np.ones((self.batch_size, self.batch_size), dtype=bool)
        if self.drop_afl:
            sample_mask[self.tgt_batch_size:, self.tgt_batch_size:] = 0 
        else:
            sample_mask[self.tgt_batch_size:, self.tgt_batch_size:] = (tgt_sample_list.reshape(-1,1) == tgt_sample_list) 
        label_mask = (label_list.reshape(-1,1) == label_list)
        pos_mask = (sample_mask & label_mask) ^ np.eye(self.batch_size, dtype=bool)
        neg_mask = sample_mask & (~label_mask)

        # shuffle
        rng = torch.randperm(self.batch_size)
        return {'img':imgs[rng], 
                'pos_mask': torch.from_numpy(pos_mask.astype(np.uint8))[:, rng][rng, :],
                'neg_mask': torch.from_numpy(neg_mask.astype(np.uint8))[:, rng][rng, :]}

    def __iter__(self):
        return self

class Triplet_DataLoader(object):
    def __init__(self, dataset, class_per_batch, image_per_class):
        self.dataset = dataset
        self.class_per_batch = class_per_batch
        self.image_per_class = image_per_class
        self.batch_size = class_per_batch * image_per_class

        self.img_list = {}
        for c in np.unique(dataset.labels):
            idxs = np.nonzero(dataset.labels == c)[0]
            if len(idxs) >= image_per_class:
                self.img_list[c] = idxs
        self.unique_labels = np.array(list(self.img_list.keys()))

    def __next__(self):
        # Sample class id
        classes = np.random.choice(self.unique_labels, self.class_per_batch)
        # Sample images
        batch_idx = np.zeros(self.batch_size)
        for i, c in enumerate(classes):
            batch_idx[i*self.image_per_class:(i+1)*self.image_per_class] = \
                np.random.choice(self.img_list[c], self.image_per_class)
        # Get images
        #imgs = torch.FloatTensor(self.batch_size, 3, 224, 224)
        imgs = []
        labels = [] #torch.LongTensor(self.batch_size)
        for i, idx in enumerate(batch_idx):
            data = self.dataset[int(idx)]
            imgs.append(data['img'])
            labels.append(data['class'])
        imgs = torch.stack(imgs, dim=0)
        labels = torch.LongTensor(labels)
        # Shuffle images and labels
        rng = torch.randperm(self.batch_size)
        return {'img':imgs[rng], 'class':labels[rng]}

    def __iter__(self):
        return self

class PCTD_DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.n_ids_list = dataset.get_n_ids_list()
        self.n_cams = len(self.n_ids_list)
        self.n_sample_per_cam = batch_size // self.n_cams

    def __next__(self):
        imgs = []
        labels = []
        for cam_id in range(self.n_cams):
            cam_id += 1
            for idx in np.random.choice(len(self.dataset.label_dict[cam_id]), size=self.n_sample_per_cam, replace=False):
                data = self.dataset.getitem(cam_id, idx)
                imgs.append(data['img'])
                labels.append(data['class'])

        imgs = torch.stack(imgs, dim=0)
        labels = torch.LongTensor(labels)
        return {'img':imgs, 'class':labels}

    def __iter__(self):
        return self

##### High-level call: Get dataloader#####
def Get_normal_DataLoader(db_txt, transform, image_per_class=0, batch_size=128, shuffle=False, num_workers=6):
    dataset = Image_Dataset(db_txt, transform, image_per_class)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def Get_unsupervised_triplet_DataLoader(pkl_name, transform, max_batch_size, mode='unfix', image_per_class=None):
    dataset = Unsupervised_Triplet_Image_Dataset(pkl_name, transform)
    return Unsupervised_Triplet_Image_DataLoader(dataset, max_batch_size, mode, image_per_class)

def Get_CDM_triplet_DataLoader(db_txt, pkl_name, transform, batch_size, image_per_class=None, drop_afl=False):
    src_dataset = Image_Dataset(db_txt, transform, image_per_class)
    tgt_dataset = Unsupervised_Triplet_Image_Dataset(pkl_name, transform)
    return CDM_Triplet_Image_DataLoader(src_dataset, tgt_dataset, batch_size, image_per_class, drop_afl)

def Get_triplet_DataLoader(db_txt, transform, class_per_batch, image_per_class):
    dataset = Image_Dataset(db_txt, transform, image_per_class)
    return Triplet_DataLoader(dataset, class_per_batch, image_per_class)

def Get_PCTD_DataLoader(db_txt, transform, batch_size=128):
    dataset = PCTD_Dataset(db_txt, transform)
    n_ids_list = dataset.get_n_ids_list()
    return PCTD_DataLoader(dataset, batch_size=batch_size), n_ids_list
