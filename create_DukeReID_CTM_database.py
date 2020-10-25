import os
import argparse
import numpy as np
from pathlib import Path
import pickle
from scipy.misc import imread
from scipy.io import loadmat
from collections import defaultdict

import utils

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_images(db_txt):
    """Read image data, extract all image info"""
    db = utils.parse_db(db_txt, ['img', 'label', 'cam', 'frame_id', 'track_id'])
    return db 

def build_tracks(db):
    """Build track dict, each dict contain its image"""
    track_dict = defaultdict(list)
    for i, img in enumerate(db['img']):
        id = (db['cam'][i], db['label'][i], db['track_id'][i])
        track_dict[id].append(img)
    return track_dict

def sample_tracks(info, track_dict, sample_interval=50):
    """Sample ctm examples at specific interval"""
    tracks = []
    for cam, label, track_id in track_dict:
        mask = (info['label'] == label) & (info['track_id'] == track_id)
        frames = info['frame_id'][mask]
        start_frame = np.min(frames)
        end_frame = np.max(frames)
        tracks.append([label, cam, start_frame, end_frame, track_id])
    tracks = np.array(tracks)
    print('total tracks:', tracks.shape[0])

    l = []
    for cam in np.unique(info['cam']):
        t = tracks[tracks[:,1]==cam, :]
        if len(t) == 0: continue

        min_frame_id = np.ceil(np.min(t[:,2]) / sample_interval) * sample_interval
        max_frame_id = np.floor(np.max(t[:,3]) / sample_interval) * sample_interval
        
        for f in range(int(min_frame_id), int(max_frame_id), sample_interval):
            select_tracks = t[(t[:,2] <= f) & (t[:,3] >= f)]
            if len(select_tracks) != 0:
                l.append([(x[1], x[0], x[4]) for x in select_tracks])
    return l

if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser(description='Database generator for CTM')
    parser.add_argument('input', help='Market database txt')
    parser.add_argument('--output_pkl', required=True, help='output pkl listing all database imgs and its label')
    parser.add_argument('--sample_interval', type=int, help='interval between mining CTM samples')
    args = parser.parse_args()

    info = read_images(args.input)
    # 1. Build track to img list, generate track idx
    track_dict = build_tracks(info)
    # 2. Sample CTM examples
    ctm_samples = sample_tracks(info, track_dict, args.sample_interval)

    neg_included_samples = 0
    for sample in ctm_samples:
        if len(sample) > 1:
            neg_included_samples += 1
    print('Negetive included samples: %d/%d (%.3f)' % \
          (neg_included_samples, len(ctm_samples), neg_included_samples/len(ctm_samples)))

    # Check
    for samples in ctm_samples:
        for track in samples:
            if track not in track_dict:
                sys.exit('WTF!')
    
    with open(args.output_pkl, 'wb') as f:
        pickle.dump({'afl_samples':ctm_samples, 'track_dict':track_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)


