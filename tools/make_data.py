#!/usr/bin/env python3
# Author: Nikolai Yakovenko
"""
Given glob of:
* mp4 vids
* csv tags (multiple files)

Combine with Pandas. Make one file with vid path and useful labels
"""

import numpy as np
import torch
import tqdm
import pandas as pd
import argparse
import glob
import os
import pathlib


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vid_path", type=str, default='/home/ubuntu/data/bball/edge-100/clips/*mp4',
        help="(glob) path to video clips")
    parser.add_argument("--data_path", type=str, default='/home/ubuntu/data/bball/edge-100/csv/*csv',
        help="(glob) path to csv of info about video clips")
    parser.add_argument("--out_path", type=str, default='/home/ubuntu/data/bball/edge-100/all_vids.csv',
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
    missing_vids = all_vids.difference(found_vids)
    print('Missing %d vids (video but not record)' % len(missing_vids))
    print(missing_vids)

    # Extract columns of interest
    SAVE_COLS = ['filepath', 'pitchType', 'Is Strike', 'speed', 'spin', 'trueSpin', 'spinEfficiency']
    merge_df = merge_df[SAVE_COLS]


    # Save results...
    print('Saving results %s to %s' % (str(merge_df.shape), args.out_path))
    merge_df.to_csv(args.out_path, index=False)





if __name__ == "__main__":
    main()