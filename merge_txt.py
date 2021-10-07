from glob import glob
import os
import numpy as np

root = '/HDD/accident_anticipation/Data/DoTA_P650N700/Test_Bbox'

tr_vids = glob(os.path.join(root, 'train/*/'))
val_vids = glob(os.path.join(root, 'val/*/'))
tr_vids.sort()
val_vids.sort()

test_vids = glob(os.path.join(root, '*/'))
test_vids.sort()

def merge(vid_path):
    files = glob(os.path.join(vid_path, '*.txt'))
    files.sort()

    new_file = vid_path + '.txt'
    new_file = new_file.replace('/.', '.')

    merged = []

    for txt in files:
        n_frame = float(int(os.path.split(txt)[1].replace('.txt', '')))
        with open(txt, 'r') as f:
            for line in f:
                _, cx, cy, w, h, _ = line.split(' ')
                cx = float(cx)
                cy = float(cy)
                w = float(w)
                h = float(h)
                write = '{:.3f}, -1.000, {:.3f}, {:.3f}, {:.3f}, {:.3f}, 0.5, -1.000, -1.000, -1.000\n'.format(n_frame, cx, cy, w, h)
                merged.append(write)

    with open(new_file, 'w') as f:
        f.writelines(merged)

for path in test_vids:
    merge(path)

"""for path in tr_vids:
    merge(path)

for path in val_vids:
    merge(path)"""