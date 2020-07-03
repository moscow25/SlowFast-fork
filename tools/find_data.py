#!/usr/bin/env python3
# Author: Nikolai Yakovenko
"""
Find vid files in directory based on CSV -- copy to new directory.
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
import pickle
import shutil

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vid_path", type=str, default='/Users/nikolai/Box/DCIM-OLD/*mov',
        help="(glob) path to video clips")
    parser.add_argument("--data_path", type=str, default='/Users/nikolai/AWS/bball/edge-100/csv/all_dfv3_hand.csv',
        help="(glob) path to csv of info about video clips")
    parser.add_argument("--out_path", type=str, default='/Users/nikolai/AWS/bball/edge-100/vids-hifi/',
        help="path to output file (as CSV)")
    args = parser.parse_args()
    print(args)

    # Load all video from path
    vids = glob.glob(args.vid_path)
    print('Found %d vid files in path %s' % (len(vids), args.vid_path))
    print(vids[:5])

    # Load all data from path
    data_files = glob.glob(args.data_path)
    print('Found %d data files in path %s' % (len(data_files), args.data_path))
    print(data_files[:5])
    data_df = [pd.read_csv(f) for f in data_files]

    # Merge all data tables
    stats_df = pd.concat(data_df)
    print('Merged data shape %s' % (str(stats_df.shape)))

    # Debug what's inside
    print('Keys & unique values count')
    print(stats_df.keys())
    print([(k,len(stats_df[k].unique())) for k in stats_df.keys()])
    #print(stats_df.head)

    # Intersect -- collect only data with video
    vid_df = pd.DataFrame(vids, columns=['filepath'])
    vid_df['name_canon'] = vid_df['filepath'].apply(lambda x: pathlib.Path(x).stem)
    #print(vid_df)
    #print(stats_df['VideoFile'].unique())
    # Watch out for missing values:
    stats_df['name_canon'] = stats_df['VideoFile'].apply(lambda x: pathlib.Path(x).stem if isinstance(x, str) else 'NONE')
    # Innor join -- only values in both tables.
    merge_df = pd.merge(vid_df, stats_df, on=['name_canon', 'name_canon'])
    print('--------------\nMerged table shape %s' % str(merge_df.shape))
    print(merge_df.keys())
    print([(k,len(merge_df[k].unique())) for k in merge_df.keys()])

    # Debug -- how many entries are missing for videos?
    found_vids = set(merge_df['name_canon'].values.tolist())
    all_vids = set(vid_df['name_canon'].values.tolist())
    all_recs = set(stats_df['name_canon'].values.tolist())
    missing_recs = all_vids.difference(found_vids)
    missing_vids = all_recs.difference(found_vids)
    print('Matched %d vids (video & record)' % len(found_vids))
    print(list(found_vids)[:10])
    print('Missing %d recs (video but no record)' % len(missing_recs))
    print(list(missing_recs)[:10])
    print('Missing %d vids (record but no video)' % len(missing_vids))
    print(list(missing_vids)[:10])

    extension = '.mov'
    # Copy the "matched" videos to new folder
    matched_vids = vid_df[vid_df['name_canon'].isin(found_vids)]
    print(matched_vids.shape)
    print(matched_vids.sample(frac=1.)['filepath'].head(n=20))
    for fp in tqdm.tqdm(matched_vids['filepath'], total = matched_vids.shape[0]):
        fpr = pathlib.Path(fp).stem
        print(fp, fpr)
        src = fp
        dst = args.out_path + fpr + extension
        shutil.copy(src, dst)




if __name__ == "__main__":
    main()