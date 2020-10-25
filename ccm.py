import math
import os
import sys
import argparse
import torch
import torchvision.transforms as T
from collections import defaultdict
import pickle
from tqdm import tqdm
import numpy as np

import utils
import inference
import cmc
import models

def get_market_cam_seq_dict():
    """Generate hash_cam for Market """
    seqs = [6, 3, 3, 6, 3, 4]
    output = {}
    k = 0
    for i, n_seq in enumerate(seqs):
        for j in range(n_seq):
            output[(i+1, j+1)] = k
            k += 1
    return output

def hash_cam_seq(cam, seq_id, cam_seq_dict):
    """Look up (cam, seq_id)"""
    output = []
    for c, s in zip(cam, seq_id):
        output.append(cam_seq_dict[(c,s)])
    return np.array(output)

def get_track_uuid(labels, track_ids):
    """Get the global uuid for (label, track_id)"""
    return labels * 1e4 + track_ids

def build_img_db(db_txt, model_path, cam_seq_dict=None):
    """Build image db"""
    if args.target == 'Market':
        db = utils.parse_db(db_txt, ['img', 'cam', 'frame_id', 'track_id', 'label', 'seq_id'])
        db['hash_cam'] = hash_cam_seq(db['cam'], db['seq_id'], cam_seq_dict)
    else: 
        db = utils.parse_db(db_txt, ['img', 'cam', 'frame_id', 'track_id', 'label'])
        db['hash_cam'] = db['cam']
    db['track_uuid'] = get_track_uuid(db['label'], db['track_id'])
    db['start_frame'] = db['frame_id']
    db['end_frame'] = db['frame_id']
    db['feature'] = inference.inference_db(db_txt, model_path)
    db['id'] = db['label']
    return db

def build_track_db(db, track_dict, cam_seq_dict=None):
    """Build track db according to track_dict"""
    if args.target == 'Market':
        track_db = {k:[] for k in ['cam', 'label', 'track_id', 'feature', 'start_frame', 'end_frame', 'seq_id']}
    else:
        track_db = {k:[] for k in ['cam', 'label', 'track_id', 'feature', 'start_frame', 'end_frame']}
    for (cam, label, track_id), imgs in track_dict.items():
        track_db['cam'].append(cam)
        track_db['label'].append(label)
        track_db['track_id'].append(track_id)
        mask = (db['label'] == label) & (db['track_id'] == track_id)
        track_db['start_frame'].append(np.min(db['frame_id'][mask]))
        track_db['end_frame'].append(np.max(db['frame_id'][mask]))
        track_db['feature'].append(np.mean(db['feature'][mask, :], axis=0))
        if args.target == 'Market':
            track_db['seq_id'].append(int(imgs[0].split('/')[-1].split('_')[1][3]))
    track_db = {k: np.array(v) for k,v in track_db.items()}
    track_db['id'] = track_db['label']
    track_db['track_uuid'] = get_track_uuid(track_db['label'], track_db['track_id'])
    if args.target == 'Market':
        track_db['hash_cam'] = hash_cam_seq(track_db['cam'], track_db['seq_id'], cam_seq_dict)
    else:
        track_db['hash_cam'] = track_db['cam']
    return track_db

def get_visual_threshold(pkl, db, low_p):
    """determine visual threshold for building travel model"""
    with open(pkl, 'rb') as f:
        ctm_samples = pickle.load(f)['afl_samples']

    dist = cmc.sqdist(db['feature'], db['feature'])
    pos_mask = np.zeros_like(dist, dtype=bool)
    neg_mask = np.zeros_like(dist, dtype=bool)
    for sample in tqdm(ctm_samples):
        for i in range(len(sample)):
            track_uuid = get_track_uuid(sample[i][1], sample[i][2])
            mask1 = (db['track_uuid'] == track_uuid)
            pos_mask[mask1, :] = mask1
            for j in range(i+1, len(sample)):
                track_uuid = get_track_uuid(sample[j][1], sample[j][2])
                mask2 = (db['track_uuid'] == track_uuid)
                neg_mask[mask1, :] = mask2
    
    return np.percentile(dist[pos_mask], 99), np.percentile(dist[neg_mask], low_p)

def build_delta_stats(db, visual_threshold, n_cams, abs):
    """Build delta statistics"""
    pos_delta_stats = [[] for i in range(n_cams)] # N(c_i, c_j, delta_i_j, S_i == S_j)
    for i in range(n_cams):
        for j in range(n_cams):
            print('processing cam (%d, %d)' % (i+1,j+1))
            mask1 = (db['hash_cam'] == i+1)
            mask2 = (db['hash_cam'] == j+1)
            dist = cmc.sqdist(db['feature'][mask1], db['feature'][mask2])
            pos_mask = dist < visual_threshold
            if i == j:  # Need to mask out duplicate entries when i == j
                unique_mask = (np.arange(dist.shape[0]) > np.arange(dist.shape[0]).reshape(-1,1))
            else:
                unique_mask = np.ones_like(dist).astype(bool)
            delta = (db['start_frame'][mask1].reshape(-1,1) - db['end_frame'][mask2])
            if abs: delta = np.abs(delta)
            pos_delta_stats[i].append(delta[pos_mask & unique_mask])

    return pos_delta_stats

def build_delta_prob(delta_stats, n_cams, max_delta, delta_bin_size):
    """Convert statistics to probability"""
    print('build delta prob...')
    delta_prob = np.zeros((n_cams, n_cams, 2*max_delta // delta_bin_size))
    bins = np.arange(-max_delta, max_delta, step = delta_bin_size)

    for i in range(n_cams):
        total_counts = np.sum([len(x) for x in delta_stats[i]])
        if total_counts == 0: total_counts = 1
        for j in range(n_cams):
            hist, _ = np.histogram(delta_stats[i][j], bins=bins)
            delta_prob[i, j, :] = hist / total_counts
    return delta_prob, bins

def get_fusion_score(q_db, g_db, visual_prob, pos_delta_prob, bins, max_delta, abs):
    """Calculate fusion score for each video pair in db"""
    print('get fusion score...')
    # generate delta_id 
    delta = (q_db['start_frame'].reshape(-1,1) - g_db['end_frame'])
    if abs: delta = np.abs(delta)
    delta_id = np.digitize(delta, bins) - 1

    # generate camera_id
    cam1 = np.tile(q_db['hash_cam'].reshape(-1, 1), (1, len(g_db['hash_cam']))) - 1
    cam2 = np.tile(g_db['hash_cam'], (len(q_db['hash_cam']), 1)) - 1

    # index pre-calculate delta probability
    pos_prob = pos_delta_prob[cam1, cam2, delta_id]

    max_delta_mask = (delta >= max_delta) | (delta <= -max_delta)
    pos_prob[max_delta_mask] = 0

    # calculate fusion probability
    st_prob = pos_prob
    fusion_score = visual_prob * st_prob
    return fusion_score

def to_rank(m, axis):
    """Rank matrix along the axis"""
    output = np.empty_like(m)
    if axis == 0:
        for i in range(m.shape[0]):
            temp = np.argsort(m[i, :])
            output[i, temp] = np.arange(len(temp))
    elif axis == 1:
        for i in range(m.shape[1]):
            temp = np.argsort(m[:, i])
            output[temp, i] = np.arange(len(temp))
    return output

def img_to_track_score(img_db, track_db, fusion_score):
    """Convert image score to track score"""
    print('convert image score to track score...')
    # Permute image index to match tracks
    permute_idx = []
    split_idx = []
    n = 0
    for i, track_uuid in enumerate(track_db['track_uuid']):
        img_idx = np.where(img_db['track_uuid'] == track_uuid)
        permute_idx += img_idx[0].tolist()
        n += len(img_idx[0])
        split_idx.append(n)
    img_score = fusion_score[permute_idx, :][:, permute_idx]
    split_idx = split_idx[:-1]

    # Compute score
    temp = np.split(img_score, split_idx, axis=0)
    temp = [np.split(x, split_idx, axis=1) for x in temp]
    return np.array([[np.mean(y) for y in x] for x in temp])

def pairs_lookup(db, pairs):
    """Look up (cam, label, id) for pairs"""
    output = []
    for i, j in np.array(pairs).T:
        key1 = (db['cam'][i], db['label'][i], db['track_id'][i])
        key2 = (db['cam'][j], db['label'][j], db['track_id'][j])
        output.append([key1, key2])
    return output

def get_samples(args, db, fusion_score, visual_score, k, v_low, v_high, n, sample_mode):
    """Get positive & negative samples"""
    print('get pos & neg samples...')
    log_median = np.log(np.median(fusion_score + np.exp(-40)))
    print('ln(median) of fusion score:', log_median)
    # pos pairs
    score = np.copy(fusion_score)
    pos_pairs = []
    pos_pair = [0]
    i = 0
    if args.no_pbbp: max_iter = 1
    else: max_iter = 10
    while (len(pos_pair) != 0) and (i < max_iter):
        col_ranking = to_rank(-1*score, 1)
        mask1 = (col_ranking <= (k-1))
        mask2 = (col_ranking.T <= (k-1))
        mask3 = np.arange(score.shape[0]) > np.arange(score.shape[0]).reshape(-1,1)
        pos_mask = (mask1 & mask2 & mask3)
        pos_pair = pairs_lookup(db, np.where(pos_mask))
        pos_pairs += pos_pair
        score[np.where(pos_mask)] = 0
        i += 1

    # neg pairs
    mask3 = np.arange(fusion_score.shape[0]) > np.arange(fusion_score.shape[0]).reshape(-1,1)
    mask4 = (np.log(fusion_score + np.exp(-40)) <= log_median)
    mask5 = (visual_score < v_high) & (visual_score > v_low)
    if sample_mode == 'v_th':
        neg_mask = mask3 & mask4 & mask5
        neg_pairs = pairs_lookup(db, np.where(neg_mask))
    elif sample_mode == 'n_neg':
        neg_mask = mask3 & mask4 
        # importance selection
        neg_pairs = np.where(neg_mask)
        if args.no_hnm:
            rand_idx = np.random.permutation(len(neg_pairs[0]))
            neg_pairs = tuple(x[rand_idx] for x in neg_pairs)
        else:
            importance = visual_score[neg_mask]
            sort_idx = np.argsort(-1*importance)
            neg_pairs = tuple(x[sort_idx] for x in neg_pairs)
        neg_pairs = tuple(x[:n*len(pos_pairs)] for x in neg_pairs)
        neg_pairs = pairs_lookup(db, neg_pairs)
    
    return pos_pairs, neg_pairs

def find_recursive(key, pairs, visit):
    """Auxiliary function for aug_pos_pairs()"""
    temp = {key} 
    for i, (key1, key2) in enumerate(pairs):
        if not visit[i]:
            if key == key1:
                visit[i] = True
                temp = temp | find_recursive(key2, pairs, visit)
            elif key == key2:
                visit[i] = True
                temp = temp | find_recursive(key1, pairs, visit)
    return temp

def aug_pos_pairs(pairs):
    """Augment pos pairs by checking"""
    visit = [False] * len(pairs)
    unique_tracks = []
    for i, (key1, key2) in enumerate(pairs):
        if not visit[i]:
            visit[i] = True
            temp = find_recursive(key1, pairs, visit)
            temp = temp | find_recursive(key2, pairs, visit)
            unique_tracks.append([k for k in temp])

    return unique_tracks

def write_samples(pkl, save_pkl, pos_pairs, neg_pairs, delete_ctm=False):
    """Write cross camera samples to pkl"""
    print('write samples to pkl')
    # Load pickle
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    track_dict = data['track_dict']
    afl_samples = data['afl_samples']

    # Add negative samples
    if delete_ctm:
        afl_samples = neg_pairs
    else:
        afl_samples += neg_pairs

    # Add positive samples
    for keys in pos_pairs:
        img_list = []
        for k in keys:
            img_list += track_dict[k]
        for k in keys:
            track_dict[k] = img_list

    # Sanity check
    for samples in afl_samples:
        for track in samples:
            if track not in track_dict:
                sys.exit('wtf!')

    with open(save_pkl, 'wb') as f:
        pickle.dump({'afl_samples':afl_samples, 'track_dict':track_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='train database txt')
    parser.add_argument('--query', required=True, help='query database txt')
    parser.add_argument('--gallery', required=True, help='gallery database txt')
    parser.add_argument('--train_pkl', required=True, help='CTM pkl')
    parser.add_argument('--model', required=True, help='CNN model')
    parser.add_argument('--save_pkl', default=None, help='path to save CTM+CCM pkl')
    parser.add_argument('--a', type=float, default=0.1, help='parameter "a"')
    parser.add_argument('--k', type=int, default=1, help='parameter "k" for mutual k-NN')
    parser.add_argument('--target', type=str, default='dukesync', choices=['Market', 'dukesync', 'DukeReID'], help='target dataset')
    parser.add_argument('--n', type=int, default=100, help='# of neg pairs (factors of # of pos pairs)')
    parser.add_argument('--low_p', type=float, default=0.1, help='percentile of vl_low')
    parser.add_argument('--sample_mode', type=str, choices=['v_th', 'n_neg'], default='n_neg', help='CCM sample mode')
    parser.add_argument('--delete_ctm', action='store_true', help='delete CTM samples')
    parser.add_argument('--abs', action='store_true', help='use abs() for delta')
    parser.add_argument('--no_hnm', action='store_true', help='use random sampling instead of hard sampling for negative mining')
    parser.add_argument('--no_pbbp', action='store_true', help='use vanilla pbbp')
    args = parser.parse_args()
    if args.target == 'Market':
        n_cams = 25
        cam_seq_dict = get_market_cam_seq_dict()
        delta_bin_size = 250
    elif args.target == 'dukesync' or args.target == 'DukeReID':
        n_cams = 8
        cam_seq_dict = None
        delta_bin_size = 600
    max_delta = int(1e+6)+1

    # Parse database & Inference training set
    train_db = build_img_db(args.train, args.model, cam_seq_dict)
    query_db = build_img_db(args.query, args.model, cam_seq_dict)
    gallery_db = build_img_db(args.gallery, args.model, cam_seq_dict)
    
    # Build video database
    with open(args.train_pkl, 'rb') as f:
        track_dict = pickle.load(f)['track_dict']
    train_track_db = build_track_db(train_db, track_dict, cam_seq_dict)
     
    # Evaluation with CNN features
    print('-----Evaluation with CNN features-----') 
    CMC, mAP = cmc.Self_Cmc_dict(train_db, rank_size=100)
    print('[Train][Image] Visual classifier CMC: %.3f %.3f %.3f %.3f %.3f, mAP: %.3f' % (CMC[0], CMC[4], CMC[9], CMC[14], CMC[19], mAP))
    if args.target == 'dukesync':
        CMC, mAP = cmc.Self_Cmc_dict(gallery_db, rank_size=100)
    else:
        CMC, mAP = cmc.Vanilla_Cmc_dict(query_db, gallery_db, rank_size=100)
    print('[Val][Image] Visual classifier CMC: %.3f %.3f %.3f %.3f %.3f, mAP: %.3f' % (CMC[0], CMC[4], CMC[9], CMC[14], CMC[19], mAP))
    print('--------------------------------------')
    
    # Determine visual threshold for building travel model
    max_pos_vdist, min_neg_vdist = get_visual_threshold(args.train_pkl, train_db, args.low_p)
    print('max_pos_vdist:', max_pos_vdist, 'min_neg_vdist:', min_neg_vdist)
     
    # Build delta distributions
    pos_delta_stats = build_delta_stats(train_db, min_neg_vdist, n_cams, args.abs)
    
    # Exclude cam_i to cam_i pair (returning event)
    if args.target == 'DukeReID' or args.target == 'dukesync':
        for i in range(n_cams):
            pos_delta_stats[i][i] = np.empty(0)
    elif args.target == 'Market': 
        temp = [0,6,9,12,18,21,25]
        for k in range(len(temp)-1):
            for i in range(temp[k], temp[k+1]):
                for j in range(i, temp[k+1]):
                    pos_delta_stats[i][j] = np.empty(0)
                    pos_delta_stats[j][i] = np.empty(0)
    
    # Convert delta distribution to probability
    pos_delta_prob, bins = build_delta_prob(pos_delta_stats, n_cams, max_delta, delta_bin_size)
  
    # Calculate fusion score
    train_visual_prob = np.exp(-1*args.a*cmc.sqdist(train_db['feature'], train_db['feature']))
    train_fusion_score = get_fusion_score(train_db, train_db, train_visual_prob, pos_delta_prob, bins, max_delta, args.abs)
    val_visual_prob = np.exp(-1*args.a*cmc.sqdist(query_db['feature'], gallery_db['feature']))
    val_fusion_score = get_fusion_score(query_db, gallery_db, val_visual_prob, pos_delta_prob, bins, max_delta, args.abs)
    
    # Evaluate fusion score
    print('-----Evaluation with Fusion prob-----') 
    CMC, mAP = cmc.Self_Cmc_dict(train_db, rank_size=1000, dist_mat= -1 * train_fusion_score)
    print('[Train][Image] Fusion model CMC: %.3f %.3f %.3f %.3f %.3f, mAP: %.3f' % (CMC[0], CMC[4], CMC[9], CMC[14], CMC[19], mAP))
    if args.target == 'dukesync':
        CMC, mAP = cmc.Self_Cmc_dict(gallery_db, rank_size=1000, dist_mat = -1 * val_fusion_score)
    else:
        CMC, mAP = cmc.Vanilla_Cmc_dict(query_db, gallery_db, rank_size=1000, dist_mat = -1 * val_fusion_score)
    print('[Val][Image] Fusion model CMC: %.3f %.3f %.3f %.3f %.3f, mAP: %.3f' % (CMC[0], CMC[4], CMC[9], CMC[14], CMC[19], mAP))
    print('-------------------------------------')
    
    if args.save_pkl is not None:
        # Mine CCM examples on train set
        train_track_visual_prob = img_to_track_score(train_db, train_track_db, train_visual_prob)
        train_track_fusion_score = img_to_track_score(train_db, train_track_db, train_fusion_score)
        pos_pairs, neg_pairs = get_samples(args, train_track_db, train_track_fusion_score, train_track_visual_prob, 
            k=args.k, v_low=np.exp(-args.a*max_pos_vdist), v_high=np.exp(-args.a*min_neg_vdist), 
            n=args.n, sample_mode=args.sample_mode)

        pos_pairs = aug_pos_pairs(pos_pairs)
        
        # Write the samples to pkl
        write_samples(args.train_pkl, args.save_pkl, pos_pairs, neg_pairs, args.delete_ctm)
    
