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
from sklearn.preprocessing import normalize, StandardScaler
import argparse
import glob
import os
import pathlib
import pickle
import hashlib


pi = 3.14159265358979323846
def deg2rad(t):
    return t * pi / 180.

def rad2deg(r):
    return r * 180. / pi

# For debugging, return degrees on clock from (x,y) as above
def tilt_from_xy(in_t, use_tanh=False, debug=False):
    if use_tanh:
        in_t = torch.tanh(in_t)
    if debug:
        print(in_t)
    s = torch.sum((in_t * in_t), dim=1)
    if debug:
        print(s)
    in_t = in_t / s.unsqueeze(1)
    rad = torch.atan2(in_t[:,1], in_t[:,0])
    return rad2deg(rad)

# Translate clock (10:00) to 0-360 degrees
def degrees_from_clock(clock):
    h, m = clock.split(':')
    h, m = int(h), int(m)
    t = 60 * h + m
    d =  360. * t / (12 * 60.)
    return d

# adjust image scale
XSCALE_BIG = 864.
XSCALE_ADJ = 320.
YSCALE_BIG = 688.
YSCALE_ADJ = 256.
def adjust_scale(x, in_scale, out_scale, is_x=False):
    # NOTE: input scale has (0,0) as TOP LEFT. Not sure that's how other scale does it...
    # we *assume* this is the case for FMPEG also.
    z = x * (out_scale/in_scale)
    return z

FIXED_SORT = True
PAD_XY = 0.5 # How far to pad the ball location on both sides, in each dimension? [for inclusion]

# pad both sides
def adjust_min_max(x, xy='x'):
    #print(x, xy)
    if xy == 'x':
        x1, x2 = x['x1_adj'], x['x2_adj']
    elif xy == 'y':
        x1, x2 = x['y1_adj'], x['y2_adj']
    min_x, max_x, delt_x = min(x1,x2), max(x1,x2), abs(x1-x2)
    #print(min_x, max_x, delt_x)
    min_x = max(min_x - PAD_XY*delt_x, 0)
    max_x = min(max_x + PAD_XY*delt_x, XSCALE_BIG if xy=='x' else YSCALE_BIG)
    return [min_x, max_x]

def main():
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument("--vid_path", type=str, default='/data/bball/edge-100/full_clips/*mov',
    #    help="(glob) path to video clips")
    parser.add_argument("--vid_path", type=str, default='/data/bball/edge-100/clips/*mp4',
        help="(glob) path to video clips")
    parser.add_argument("--data_path", type=str, default='/home/ubuntu/data/bball/edge-100/csv/all_dfv3_hand_exclude.csv',
        help="(glob) path to csv of info about video clips")
    parser.add_argument("--out_path", type=str, default='/home/ubuntu/open_source/SlowFast-fork/data/edge-100/',
        help="path to output file (as CSV)")
    parser.add_argument('--val_frac', type=float, default=0.3333,
        help='validation split?')
    parser.add_argument('--adjust_scale', action='store_true', default=False,
        help='adjust scale for rescaled video? (XY coordinates, etc)')
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
    print('Missing %d vids (video but no record)' % len(missing_vids))
    print(missing_vids)

    # Extract columns of interest
    SAVE_COLS = ['filepath', 'name_canon', 'fullName', 'ReleaseFrame', 'pitchType', 'Is Strike', 'speed', 'spin', 'trueSpin', 'spinEfficiency',
        'Top Spin', 'Side Spin', 'Rifle Spin', 'spinAxis', 'vb','hb', 'Horizontal Angle', 'Release Angle',
        'arm_angle', 'arm_angle_class', 'Handedness', 'x1_new', 'y1_new', 'x2_new', 'y2_new', 'isError']
    # TODO: Store the bro's name -- for eval purposes
    # TODO: Who's a lefty? Also -- flip images in training...
    print(merge_df.keys())
    #assert(False)

    merge_df = merge_df[SAVE_COLS]

    #print(merge_df.head())
    #print(merge_df.tail())
    #print(merge_df['spinAxis'])

    # SpinAxis is special -- 1) translate from "11:30" to degrees 0-360 2) custom loss to handle discontinuity at 0
    merge_df['spinAxisDeg'] = merge_df['spinAxis'].apply(lambda x: degrees_from_clock(x))

    # Handedness -- "is lefty"
    merge_df['isLefty'] = merge_df['Handedness'].apply(lambda x: 1.0 if x.upper() == 'L' else 0.)
    merge_df['isStrike'] = merge_df['Is Strike'].apply(lambda x: 1.0 if x.upper() == 'YES' else 0.)

    # Debug -- notice the patterns in pitch type, spin axis and break... guys are pretty consistent about this.
    print(merge_df[['pitchType', 'speed', 'spin', 'vb','hb', 'spinAxis', 'spinAxisDeg', 'isLefty']])

    # Normalize stats for regression. [Save norm values so we can rebuild projects]
    cols_to_normalize = ['speed', 'spin', 'trueSpin', 'spinEfficiency',
        'Top Spin', 'Side Spin', 'Rifle Spin', 'vb','hb', 'Horizontal Angle', 'Release Angle', 'arm_angle', 'isLefty', 'isStrike']
    norm_inputs = merge_df[cols_to_normalize].values

    print('Normalizing values for regression...')
    print(norm_inputs.mean(axis=0))
    print(merge_df[['pitchType', 'Is Strike', 'speed', 'spin', 'trueSpin', 'spinEfficiency']][:10])
    print(norm_inputs[:10])
    print(norm_inputs.shape)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(norm_inputs)
    norm_out = scaler.transform(norm_inputs)
    print(norm_out.shape)
    print(norm_out[:])
    print("scale, mean and var for %d features" % len(scaler.scale_))
    print(cols_to_normalize)
    print((scaler.scale_, scaler.mean_, scaler.var_))
    merge_df['speed_norm'] = norm_out[:,0]
    merge_df['spin_norm'] = norm_out[:,1]
    merge_df['trueSpin_norm'] = norm_out[:,2]
    merge_df['spinEfficiency_norm'] = norm_out[:,3]
    merge_df['topSpin_norm'] = norm_out[:,4]
    merge_df['sideSpin_norm'] = norm_out[:,5]
    merge_df['rifleSpin_norm'] = norm_out[:,6]
    merge_df['vb_norm'] = norm_out[:,7]
    merge_df['hb_norm'] = norm_out[:,8]
    merge_df['hAngle_norm'] = norm_out[:,9]
    merge_df['rAngle_norm'] = norm_out[:,10]
    merge_df['armAngle_norm'] = norm_out[:,11]
    merge_df['isLefty_norm'] = norm_out[:,12]
    merge_df['isStrike_norm'] = norm_out[:,13]
    print(merge_df.keys())
    print(merge_df.head())

    # For X1..Y2 -- bounding box on the detected ball -- translate from 900x700 coordinates (top left 0,0) to 320x256 coordinates
    merge_df['x1_adj'] = merge_df['x1_new']
    merge_df['x2_adj'] = merge_df['x2_new']
    merge_df['y1_adj'] = merge_df['y1_new']
    merge_df['y2_adj'] = merge_df['y2_new']
    # Do we pad the (x,y) first?
    if PAD_XY > 0.:
        # reset new bounds as new XY
        merge_df[['x1_adj', 'x2_adj']] = merge_df[['x1_adj', 'x2_adj']].apply(lambda x: adjust_min_max(x, xy='x'), axis=1).tolist()
        merge_df[['y1_adj', 'y2_adj']] = merge_df[['y1_adj', 'y2_adj']].apply(lambda x: adjust_min_max(x, xy='y'), axis=1).tolist()

    if args.adjust_scale:
        merge_df['x1_adj'] = merge_df['x1_adj'].apply(lambda x: adjust_scale(x, in_scale=XSCALE_BIG, out_scale=XSCALE_ADJ, is_x=True))
        merge_df['x2_adj'] = merge_df['x2_adj'].apply(lambda x: adjust_scale(x, in_scale=XSCALE_BIG, out_scale=XSCALE_ADJ, is_x=True))
        merge_df['y1_adj'] = merge_df['y1_adj'].apply(lambda x: adjust_scale(x, in_scale=YSCALE_BIG, out_scale=YSCALE_ADJ, is_x=False))
        merge_df['y2_adj'] = merge_df['y2_adj'].apply(lambda x: adjust_scale(x, in_scale=YSCALE_BIG, out_scale=YSCALE_ADJ, is_x=False))
    else:
        pass

    # Drop data without 'Release Frame'?
    #print(merge_df['Release Frame'])
    vids_count = merge_df.shape[0]
    # has release frame
    merge_df = merge_df[~merge_df['ReleaseFrame'].isna()]
    # has (x,y) for ball release
    merge_df = merge_df[~merge_df['x1_adj'].isna()]
    # not manually marked as error
    merge_df = merge_df[merge_df['isError']==0]
    print('Kept %d/%d vids with Release Frame data.' % (merge_df.shape[0], vids_count))

    # Shuffle.
    merge_df = merge_df.sample(frac=1.)

    # Sort by hash(name) -- for consistent order...
    #hash_object = hashlib.md5(b'Hello World')
    #print(hash_object.hexdigest())

    merge_df['name_hash'] = merge_df['name_canon'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    if FIXED_SORT:
        merge_df = merge_df.sort_values(by='name_hash')
        print(merge_df.head())


    # Train/Val split
    assert args.val_frac == 0.3333, 'Hack -- need 3x cross fold validation'
    val_size = int(merge_df.shape[0]*args.val_frac)
    print('Train/Val split: %d/%d' % (merge_df.shape[0]-val_size, val_size))

    # Save results...
    print('Saving results %s to %s' % (str(merge_df.shape), args.out_path))
    merge_df.to_csv(os.path.join(args.out_path, 'all_vids.csv'), index=False)

    # Create 3x version of train/val split
    merge_df[val_size:].to_csv(os.path.join(args.out_path, 'train_1_3.csv'), index=False)
    merge_df[:val_size].to_csv(os.path.join(args.out_path, 'val_1_3.csv'), index=False)
    pd.concat([merge_df[:val_size], merge_df[-val_size:]]).to_csv(os.path.join(args.out_path, 'train_2_3.csv'), index=False)
    merge_df[val_size:-val_size].to_csv(os.path.join(args.out_path, 'val_2_3.csv'), index=False)
    merge_df[:-val_size].to_csv(os.path.join(args.out_path, 'train_3_3.csv'), index=False)
    merge_df[-val_size:].to_csv(os.path.join(args.out_path, 'val_3_3.csv'), index=False)
    # Default train with first 1/3
    merge_df[val_size:].to_csv(os.path.join(args.out_path, 'train.csv'), index=False)
    merge_df[:val_size].to_csv(os.path.join(args.out_path, 'val.csv'), index=False)
    # Save normalization function (useful to transform predictions back to real value)
    with open(os.path.join(args.out_path, 'data_scaler.pkl'), 'wb') as norm_file:
        pickle.dump(scaler, norm_file)

if __name__ == "__main__":
    main()