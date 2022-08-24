import os
import glob
import numpy as np
import h5py
from joblib import Parallel, delayed
import multiprocessing
import argparse
import sys
sys.path.append('..')
from cadlib.visualize import vec2CADsolid
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
args = parser.parse_args()

valid_cad = []
invalid_cad = []
novel_cad = []
unique_cad = []


def issame(train_vec, out_vec):
    train_vec = np.array(train_vec)
    out_vec = np.array(out_vec)
    return np.array_equal(train_vec, out_vec)

def process_one(path):
    data_id = path.split("/")[-1]
    
    idx = int(data_id.split('.')[0])

    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(np.float)

    try:
        # CAD Sequence가 Valid 여부 측정
        shape = vec2CADsolid(out_vec)
        valid_cad.append(out_vec)       
        
    except Exception as e:
        print("create_CAD failed", data_id)
        invalid_cad.append(out_vec)


all_paths = glob.glob(os.path.join(args.src, "*.h5"))

train_dir = './traindata/'
train_data = []
for i in range(314):
    train_path = train_dir + '{}_vec.h5'.format(i)
    with h5py.File(train_path, 'r') as fp:
        train_data.append(fp['out_vec'][:])
train_data = np.array(train_data)
train_data = train_data.reshape(-1, 60, 17)

train_cad = []
for i in range(len(train_data)):
    cmd = train_data[i][:, 0]
    idx = np.where(cmd == 3)[0][0]
    
    train_cad.append(train_data[i][:idx])
    

with Parallel(n_jobs=8, verbose=2, require='sharedmem') as parallel:
    parallel(delayed(process_one)(x) for x in all_paths)

# novel
for i in tqdm(range(len(valid_cad))):
    is_novel = True
    for j in range(i + 1, len(valid_cad)):
        a = np.array(valid_cad[i])
        b = np.array(valid_cad[j])
        if issame(a, b):
            is_novel = False
            break
    if is_novel:
        novel_cad.append(valid_cad[i])

# unique
for i in tqdm(range(len(valid_cad))):
    is_unique = True
    for j in range(len(train_cad)):
        if issame(valid_cad[i], train_cad[j]):
            is_unique = False
            break
    if is_unique:
        unique_cad.append(valid_cad[i])

# 결과 출력
print('valid ratio: ', len(valid_cad) / (len(valid_cad) + len(invalid_cad)))
print('novel ratio: ', len(novel_cad) / len(valid_cad))
print('unique ratio: ', len(unique_cad) / len(valid_cad))