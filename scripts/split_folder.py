import argparse
from boltons import fileutils
import os.path as osp
import os
from shutil import copy
from tqdm import tqdm
import random


parser = argparse.ArgumentParser(description="split a folder with data into train/val/test")
parser.add_argument('--src', help="source folder with unsplit data files")
parser.add_argument('--trg', help="target folder, this is where all the split data will go")
parser.add_argument('--ratio', default="0.8,0.1,0.1", help="ratio of train/val/test split")
parser.add_argument('--ext', help="use this extension to find related files. for example, if searching for .wav \
    then if we have 1.wav, this script will search for all files 1.* to copy together with 1.wav")
parser.add_argument('--shuffle', type=bool, default=True, help="should the data be shuffled before copied?")
args = parser.parse_args()

os.makedirs(args.trg, exist_ok=True)
os.makedirs(osp.join(args.trg, "train"), exist_ok=True)
os.makedirs(osp.join(args.trg, "val"), exist_ok=True)
os.makedirs(osp.join(args.trg, "test"), exist_ok=True)

data = []
all_files = list(fileutils.iter_find_files(args.src, f"*"))
ext_files = list(fileutils.iter_find_files(args.src, f"*{args.ext}"))
n_examples = len(ext_files)

ratio = list(map(float, args.ratio.split(",")))
assert sum(ratio) == 1, "ratios should sum to 1!"
ratio = [ratio[0], ratio[0] + ratio[1], sum(ratio)]
train_range, val_range, test_range = int(n_examples * ratio[0]), int(n_examples * ratio[1]), int(n_examples * ratio[2])

for f in tqdm(ext_files, desc="indexing data"):
    basename = osp.basename(f).split(".")[0]
    related = set(filter(lambda x: basename in x, all_files))
    data.append(related)
if args.shuffle:
    random.shuffle(data)

for i, example in enumerate(tqdm(data, desc="copying files")):
    if i < train_range:
        target = osp.join(args.trg, "train")
    elif train_range <= i < val_range:
        target = osp.join(args.trg, "val")
    else:
        target = osp.join(args.trg, "test")
    
    for f in example:
        copy(f, target)