import numpy as np
import torch
import sys
import pandas as pd
from progressbar import ProgressBar, AnimatedMarker, Percentage
from tqdm import tqdm

import utils

def Sample_query(ids, cams):
    unique_ids = np.unique(ids)
    unique_cams = np.unique(cams)
    unique_ids = unique_ids[unique_ids>0] # id -1, 0 cannot be query

    query_idx = []
    for id in unique_ids:
        for cam in unique_cams:
            query_candidate = np.where((ids == id) & (cams == cam))[0]
            gallery_candidate = np.where((ids == id) & (cams != cam))[0]
            if len(query_candidate) > 0 and len(gallery_candidate) != 0: 
                query_idx.append(query_candidate[0])
    gallery_idx = np.arange(len(ids))
    #print('Query: %d, Gallery: %d' % (len(query_idx), len(gallery_idx)))
    return query_idx, gallery_idx

def Self_Cmc_dict(data, rank_size, dist_mat=None, dist_type='eucl'):
    """
    Perform CMC/mAP evaluation on single dataset
    data: a dict contains 'cam', 'id', 'feature' list
    """    
    # Sample query
    q_idx, g_idx = Sample_query(data['id'], data['cam'])
    q_data = {k:v[q_idx] for k, v in data.items() if v is not None}
    g_data = {k:v[g_idx] for k, v in data.items() if v is not None}
    if len(g_idx) < rank_size: rank_size = len(g_idx)
    if dist_mat is not None:
        dist_mat = dist_mat[:, g_idx]
        dist_mat = dist_mat[q_idx, :]
    CMC, mAP = Cmc(q_data, g_data, rank_size, dist_mat, dist_type)
    return CMC, mAP

def Self_Cmc(features, db_txt, rank_size, dist_mat=None, dist_type='eucl'):
    db = utils.parse_db(db_txt, ['label', 'cam'])
    data = {'feature':features, 'id':db['label'], 'cam':db['cam']}
    CMC, mAP = Self_Cmc_dict(data, rank_size, dist_mat, dist_type)
    return CMC, mAP

def Vanilla_Cmc_dict(q_data, g_data, rank_size, dist_mat=None, dist_type='eucl'):
    CMC, mAP = Cmc(q_data, g_data, rank_size, dist_mat)
    return CMC, mAP

def Vanilla_Cmc(q_features, q_db_txt, g_features, g_db_txt, rank_size, dist_mat=None, dist_type='eucl'):
    q_db = utils.parse_db(q_db_txt, ['label', 'cam'])
    g_db = utils.parse_db(g_db_txt, ['label', 'cam'])
    q_data = {'feature':q_features, 'id':q_db['label'], 'cam':q_db['cam']}
    g_data = {'feature':g_features, 'id':g_db['label'], 'cam':g_db['cam']}
    CMC, mAP = Vanilla_Cmc_dict(q_data, g_data, rank_size, dist_mat, dist_type)
    return CMC, mAP
    
def Cmc(q_data, g_data, rank_size, dist_mat=None, dist_type='eucl'):
    n_query = q_data['id'].shape[0]
    n_gallery = g_data['id'].shape[0]

    if dist_mat is not None:
        dist = dist_mat
    else:
        if dist_type == 'eucl':
            dist = sqdist(q_data['feature'], g_data['feature']) # Reture a n_query*n_gallery array
        elif dist_type == 'cos':
            dist = np_cdist(q_data['feature'], g_data['feature'])
        else:
            raise RuntimeError('Unrecognized dist type "%s"' % dist_type)
    cmc = np.zeros((n_query, rank_size))
    ap = np.zeros(n_query)
    
    #widgets = ["I'm calculating cmc! ", AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' (', Percentage(), ')']
    #pbar = ProgressBar(widgets=widgets, max_value=n_query)
    #for k in range(n_query):
    for k in tqdm(range(n_query), ncols=100):
        good_idx = np.where((q_data['id'][k]==g_data['id']) & (q_data['cam'][k]!=g_data['cam']))[0]
        junk_mask1 = (g_data['id'] == -1)
        junk_mask2 = (q_data['id'][k]==g_data['id']) & (q_data['cam'][k]==g_data['cam'])
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        score = dist[k, :]

        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:rank_size]

        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)
        #pbar.update(k)
    #pbar.finish()
    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP

def Compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n-njunk:] = 1
            flag = 1 # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue # junk image
        
        if flag == 1:
            intersect_size += 1
        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap += (recall-old_recall) * (old_precision+precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1
       
        if good_now == ngood:
            return ap, cmc
    return ap, cmc

def cdist(feat1, feat2):
    """Cosine distance"""
    feat1 = torch.FloatTensor(feat1)#.cuda()
    feat2 = torch.FloatTensor(feat2)#.cuda()
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
    dist = -1 * torch.mm(feat1, feat2)
    return dist.cpu().numpy()

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def sqdist(feat1, feat2, M=None):
    """Mahanalobis/Euclidean distance"""
    if M is None: M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)

if __name__ == '__main__':
    from scipy.io import loadmat
    q_feature = loadmat(sys.argv[1])['ff']
    q_db_txt = sys.argv[2]
    g_feature = loadmat(sys.argv[3])['ff']
    g_db_txt = sys.argv[4]
    #print(feature.shape)
    CMC, mAP = Self_Cmc(g_feature, g_db_txt, 100)
    #CMC, mAP = Vanilla_Cmc(q_feature, q_db_txt, g_feature, g_db_txt)
    print('r1 precision = %f, mAP = %f' % (CMC[0], mAP))
