#!/usr/bin/env python3
# Author: Nikolai Yakovenko
"""
Resize bunch of large .mov files -- ideally to a good aspect ratio.
"""

import numpy as np
import torch
import tqdm
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
import argparse
import glob
import os
import pathlib
import random
import pickle
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vid_path", type=str, default='/data/bball/edge-100/full_clips_Kent_0915/*mov',
        help="(glob) path to video clips")
    parser.add_argument("--out_path", type=str, default='/data/bball/edge-100/full_clips_Kent_0915_resize/',
        help="path to output file (as CSV)")
    parser.add_argument("--grayscale", action="store_true", default=False,
        help='convert vid to grayscale (dont if B&W already)')
    parser.add_argument("--threads", type=int, default=6, help="number of ffmpeg threads?")
    args = parser.parse_args()
    print(args)

    # Load all video from path
    vids = glob.glob(args.vid_path)
    print('Found %d vid files in path %s' % (len(vids), args.vid_path))
    print(vids[:5])
    random.shuffle(vids)
    max_items = 2000 # no max

    # For each video, we will run reize operation as ffmpeg command
    for v in tqdm.tqdm(vids[:max_items], total=len(vids)):
        infile = v
        infile_stem = pathlib.Path(infile).stem
        # HACK -- input should be 864x688
        #scale = "432x344"
        scale = "scale=iw/2:ih/2"
        # gray scale?
        # -vf format=gray
        if args.grayscale:
            other_params = "format=gray"
            params = scale+","+other_params
        else:
            other_params = ""
            params = scale
        format = "-pix_fmt yuv420p"
        threads = args.threads
        outfile = args.out_path + infile_stem + ".mp4"
        cmd = "ffmpeg -threads "+str(threads)+" -i "+infile+" "+format+" -vf "+params+" "+outfile
        print(cmd)
        try:
            sub_out = subprocess.run(cmd, shell=True, check=True)
            print(sub_out)
        except subprocess.CalledProcessError:
            print('Error with file: '+infile)


    print("FIN")


if __name__ == "__main__":
    main()