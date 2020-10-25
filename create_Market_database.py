import argparse
import os
import pandas as pd
from collections import OrderedDict

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('output_txt')
    parser.add_argument('--track_interval', type=int,
        help='max interval between two detections to be considered as the same track for AFL')
    args = parser.parse_args()

    # Read images
    data_dir = args.data_dir
    files = sorted(os.listdir(data_dir))
    files = [f for f in files if is_image_file(f)]
    imgs = []
    labels = []
    cams = []
    seq_ids = []
    frame_ids = []
    track_ids = []
    for i, f in enumerate(files):
        imgs.append(os.path.abspath(os.path.join(data_dir, f)))
        v = f.split('_')
        labels.append(int(v[0]))
        cams.append(int(v[1][1]))
        seq_ids.append(int(v[1][3]))
        frame_ids.append(int(v[2]))
        if i == 0 : 
            track_id = 0
        else:
            if prev_label == int(v[0]):
                if prev_camseq == v[1]:
                    if (int(v[2]) - prev_frame) < args.track_interval:
                        track_id = prev_track_id
                    else:
                        track_id += 1
                else:
                    track_id += 1
            else:
                track_id = 0
        track_ids.append(track_id)
        prev_camseq = v[1]
        prev_label = int(v[0])
        prev_frame = int(v[2])
        prev_track_id = track_id

    '''
    # Re-index id:
    # If id <=0: use original id
    # If id >0: remap to start from 1
    id_dict = {}
    idx = 1
    for id in labels:
        if id not in id_dict:
            if id <= 0:
                id_dict[id] = id
            else:
                id_dict[id] = idx
                idx += 1
    labels = [id_dict[l] for l in labels]
    '''

    # Write file
    d = OrderedDict([('img',imgs), ('label',labels), ('cam',cams), ('seq_id',seq_ids),
                     ('frame_id',frame_ids), ('track_id',track_ids)])
    df = pd.DataFrame(d)
    df.to_csv(args.output_txt, sep=' ', index=False)
