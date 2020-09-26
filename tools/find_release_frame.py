#!/usr/bin/env python3
# Author: Driveline Baseball
"""
Given videos of thrown baseball, find frame and location of the first time ball is clearly visible in a frame.
* collect candidates via CV2.findContours
* consider any square-like contours
* see if the square contains something like a baseball

PS Uses CV2 and rules; should replace with labels and ML
"""

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re
import pandas as pd
import glob
import pathlib
import tqdm

def is_baseball(img, frame=0, cnt=0, min_conf=50., scratch_dir='./', fname='', debug=False):
    # dimensions
    h = img.shape[0]
    w = img.shape[1]

    if debug:
        print('looking for baseball -- hxw '+str([h,w]))

    # convert to grayscale, and then two different binary treshholds
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, thresh = cv.threshold(gray, 0, 255, 10)
    #ret, trunc = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    # detect circles
    min_rad = int(max(h,w)/3)
    min_dist = int(max(h,w)/6)
    if debug:
        print('looking for a circle...')
        print('Looking for circle with min radius, min_dist: '+str([min_rad, min_dist]))
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=50, param2=30, minRadius=min_rad, maxRadius=int(w))

    if debug:
        print(circles)

    if not(circles is not None):
        if debug:
            print('No circles found')
        return False

    # Binary thresholds -- do we want this or not?
    ret, thresh = cv.threshold(gray, 0, 255, 10)
    ret, trunc = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)

    # Green mask
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    #mask = cv.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv.inRange(hsv, (60, 50, 50), (140, 255, 255))
    #u_green = np.array([104, 153, 70])
    #l_green = np.array([30, 30, 0])

    #mask = cv.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    # How much green in the image?
    max_full_image_green_ratio = 0.3 # careful as ball background could be green in fact... depends on camera
    green_ratio = np.sum(imask) / (h*w)
    if debug:
        print('Image has green ratio: ' + str(green_ratio))

    # Draw all circles...
    frame_w_circles = img.copy()
    circles_u = np.uint16(np.around(circles))
    for i in circles_u[0,:]:
        # draw the outer circle
        cv.circle(frame_w_circles,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(frame_w_circles,(i[0],i[1]),2,(0,0,255),3)
    cnt_file = scratch_dir + 'frame_'+str(frame)+'_cnt_'+str(cnt)+'_circles_'+fname+'.jpg'
    green_cnt_file = scratch_dir + 'frame_'+str(frame)+'_cnt_'+str(cnt)+'_green_'+fname+'.jpg'
    if debug:
        print('Drawing circles to file: '+cnt_file)
        cv.imwrite(cnt_file, frame_w_circles)
        cv.imwrite(green_cnt_file, green)

    circles = np.round(circles[0, :]).astype("int")
    # should only be one circle, otherwise probably not a baseball
    if len(circles) > 2 or green_ratio > max_full_image_green_ratio:
        return False

    # make sure something was detected -- contrast between inside and outside the circle
    #min_conf = 50. # 10. # 25. # 60.
    max_green_ratio = 0.15
    for circle in circles:
        # create mask for ball and background
        (x, y, r) = circle
        ball_mask = np.zeros((h,w), np.uint8)
        cv.circle(ball_mask,(x,y),r,(255),-1)
        bg_mask = cv.bitwise_not(ball_mask)
        # get average pixel value in ball and backrgound
        ball_avg = cv.mean(trunc, mask=ball_mask)[::-1][3]
        bg_avg = cv.mean(trunc, mask=bg_mask)[::-1][3]
        # must be large difference between ball and background + in bounds
        conf = ball_avg-bg_avg
        # NOTE: in-bounds within 5% would be fine...
        #slack = max(h,w) * 0.05 # larger slack for dark clips...
        slack = max(h,w) * 0.02 # smaller value for bright clips
        in_bounds = (x - r > 0 - slack) and (x + r < w + slack) and (y - r > 0 - slack) and (y + r < h + slack)

        # Make sure the ball is not green/green screen (probably not a ball)
        green_ball_sum = np.sum(imask * ball_mask)
        all_ball_sum = np.sum(ball_mask)
        green_ball_ratio = green_ball_sum / all_ball_sum

        if debug:
            print('conf, min_conf, in_bounds, green_ball_ratio'+str([conf, min_conf, in_bounds, green_ball_ratio]))
        if conf >= min_conf and in_bounds and green_ball_ratio < max_green_ratio:
            return (x, y, r)

    if debug:
        print('No circles found')
    return False

'''
Takes in path to video and returns the frame # at ball-release if detected, otherwise None
'''
def release_frame(vid_path, debug=False, args={}, min_frame=0, max_frame=2000): #, scratch_dir = '/data/bball/edge-100/scratch/'):
    scratch_dir = args.debug_dir
    if debug:
        print('--------------\nNew file: ', vid_path)
    vid = cv.VideoCapture(vid_path)
    fname = pathlib.Path(vid_path).stem

    # use for motion
    subtract = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    i = 0
    while i < max_frame:
        isFrame, frame = vid.read()
        if not isFrame: break

        """
        # Don't save every frame -- should have minimum evidence that frame has something in it
        if i > min_frame and debug:
            #image_file = scratch_dir + 'frame_'+str(i)+'_'+fname+'.jpg'
            image_file = scratch_dir + 'frame_'+str(i)+'.jpg'
            cv.imwrite(image_file, frame)
        """

        # dimensions
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        #convert to grayscale, blur, and detect motion
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (11, 11), 0)
        motion = subtract.apply(blurred)
        # remove noise, convert to binary, dilate
        kernel = np.ones((3,3),np.uint8)
        noiseless = cv.morphologyEx(motion, cv.MORPH_OPEN,kernel, iterations = 2)
        ret, thresh = cv.threshold(noiseless, 0, 255, 9)
        switch = cv.bitwise_not(thresh)
        kernel2 = np.ones((5,5),np.uint8)
        dilated = cv.dilate(switch, kernel2 ,iterations=4)
        #find contours in dilated image
        contours, hierarchy = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0 and debug:
            print('Frame '+str(i)+' found contours '+str(len(contours)))
        #print(contours)

        for ci, cnt in enumerate(contours):
            # must be the expected size of a baseball
            a = cv.contourArea(cnt)
            if a in range(1000,10000): # range(1000,2000): #range(4000,10000):
                x,y,w,h = cv.boundingRect(cnt)

                if debug:
                    print('Examining contour (x,y,w,h):')
                    print([x,y,w,h])
                    print('area '+str(a))

                # must be away from edge or the image, square-like, and fill up most of bounding box
                border = max(25, int(0.07*min(frame_h,frame_w)))
                in_middle = y > border and y+h < frame_h - border and x > border and x+w < frame_w - border
                square = abs(1-(min(w,h)/max(w,h))) < 0.15 # 0.075
                # *vary majority* of balls are between 60px and 150px tall -- scaling to ~650 height -- being fairly generous
                MIN_SIZE, MAX_SIZE = int(0.08*min(frame_h,frame_w)), int(0.16*min(frame_h,frame_w))
                right_size = (h+w) / 2 >= MIN_SIZE and (h+w) / 2 <= MAX_SIZE
                filled = a/(w*h) > 0.75
                if debug:
                    print('squariness '+str(abs(1-w/h)))
                    print('filledness '+str(a/(w*h)))
                    # if meets all requirements, release-frame is detected and returned
                    print('in_middle, square, right_size, filled: '+str([in_middle, square, right_size, filled]))
                    print('attempting is_baseball on this item')

                """
                # Don't save box unless it meets (generous) size and location requirements
                if i > min_frame and debug:
                    cnt_file = scratch_dir + 'frame_'+str(i)+'_cnt_'+str(ci)+'_'+fname+'.jpg'
                    cv.imwrite(cnt_file, frame[y:y+h, x:x+w])
                """

                if in_middle and square and right_size:
                    # TODO -- save full frame if we got this far??
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    if i > min_frame and debug:
                        image_file = scratch_dir + 'frame_'+str(i)+'_'+fname+'.jpg'
                        #image_file = scratch_dir + 'frame_'+str(i)+'.jpg'
                        cv.imwrite(image_file, frame)

                    if is_baseball(frame[y:y+h, x:x+w].copy(),frame=i,cnt=ci,min_conf=args.min_circ_conf,scratch_dir=scratch_dir,fname=fname, debug=debug):
                        if debug:
                            print('--------------\Found file: ', vid_path)
                            print('dims', frame_h, frame_w)

                        # Run item again, to save image...
                        if debug:
                            is_baseball(frame[y:y+h, x:x+w].copy(),frame=i,cnt=ci,min_conf=args.min_circ_conf,scratch_dir=scratch_dir,fname=fname, debug=True)

                        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                        #plt.imshow(frame)
                        #plt.show()
                        if debug:
                            print(i, x, y, x + w, y + h)

                        # Save bounding box super-imposed on the original image
                        image_file = scratch_dir + 'frame_'+str(i)+'_bbox_'+fname+'.jpg'
                        cv.imwrite(image_file, frame)

                        if debug:
                            print('ReleaseFrame ('+str(i)+') found in vid!! '+str(vid_path))

                        return i, x, y, x + w, y + h, frame_w, frame_h
                        #return i
        # not detected, next frame
        i += 1
        #print('none in frame ' +str(i))
    print('No ReleaseFrame found in vid: '+str(vid_path))
    return None

"""
dataframe_file = ''
VIDEOFOLDER_PATH = ''
df = pd.read_csv(dataframe_file)
if sum(df['VideoFile'].notnull()) == 0:
    print('No vids found')
    return

matched_videos = list(df.VideoFile)
"""

"""
#matched_videos = ['Lange_A_PitchDesign_KentFacility_20-Aug-20_1597956120.mov', ]
vids = glob.glob('/data/bball/edge-100/full_clips_Kent/*mov')
matched_videos = [pathlib.Path(f).name for f in vids]
"""

#VIDEOFOLDER_PATH = '/data/bball/edge-100/full_clips_Kent'

# Hack -- only look at a few vids in a bigger file
vids_interest = ['Thomas_N_LiveABs_Main_Gym_10-Jan-20_1578696815', 'Bauer_T_Research_07-Dec-19_1575678857',
                'Thomas_N_Live_ABs_Main_Gym_06-Jan-20_1578348550', 'Carpenter_D_LiveABs_Main_Gym_11-Jan-20_1578783719',
                'Reninger_Z_ProDay_Main_Gym_12-Jan-20_1578856777', 'Reninger_Z_ProDay_Main_Gym_12-Jan-20_1578856645',
                'Lane_T_LiveABs_Main_Gym_11-Jan-20_1578785115', 'Reninger_Z_ProDay_Main_Gym_12-Jan-20_1578856645',
                'Haymans_C_ProDay_Main_Gym_12-Jan-20_1578867805', 'Lane_T_LiveABs_Main_Gym_11-Jan-20_1578784947',
                'Carpenter_D_LiveABs_Main_Gym_11-Jan-20_1578783675', 'Mann_B_ProDay_Main_Gym_12-Jan-20_1578857480',
                'Thomas_N_LiveABs_Main_Gym_10-Jan-20_1578697515', 'Carpenter_D_LiveABs_Main_Gym_11-Jan-20_1578784112',
                'Finnegan_B_PitchDesign_Research_08-Jan-20_1578508752', 'Bauer_T_Research_07-Dec-19_1575679037',
                'Carpenter_D_PitchDesign_Research_09-Jan-20_1578596882', 'Ludeman_B_Live_ABs_Main_Gym_07-Jan-20_1578441393',
                'Jungmann_T_ProDay_Main_Gym_12-Jan-20_1578855838', 'Lane_T_LiveABs_Main_Gym_11-Jan-20_1578785033'
                'Gonzales_B_ProDay_Main_Gym_12-Jan-20_1578864532',
                'Gonzales_B_PitchDesign_Research_04-Jan-20_1578177129',
                'Stirewalt_T_ProDay_Main_Gym_12-Jan-20_1578863609',
                'Bellina_S_LiveABs_Main_Gym_11-Jan-20_1578786241',
                'Mann_B_ProDay_Main_Gym_12-Jan-20_1578857847',
                'Watts_J_ProDay_Main_Gym_12-Jan-20_1578862701',
                'Nakagawa_K_Research_09-Dec-19_1575927877',
                'Mann_B_ProDay_Main_Gym_12-Jan-20_1578857734',
                'Yoon_S_PitchDesign_Research_11-Feb-20_1581463546',
                'Nakagawa_K_Research_09-Dec-19_1575929628',
                'Gonzales_B_Live_ABs_Main_Gym_07-Jan-20_1578439408',
                'Jungmann_T_ProDay_Main_Gym_12-Jan-20_1578855878',
                'Gonzales_B_Live_ABs_Main_Gym_07-Jan-20_1578439363'
                'Bellina_S_LiveABs_Main_Gym_11-Jan-20_1578786057',
                'Carpenter_D_PitchDesign_Research_09-Jan-20_1578596341'
                'Jungmann_T_ProDay_Main_Gym_12-Jan-20_1578856240',
                ]


vids_glob = '/data/bball/edge-100/full_clips_Kent_0915/Barber_A_*.mov'
vids_interest = glob.glob(vids_glob)
vids_interest = [pathlib.Path(x).stem for x in vids_interest]

vids_interest = []

#vids_interest = ['Barber_A_PitchDesign_KentFacility_11-Sep-20_1599860804']

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default='/data/bball/edge-100/csv/09152020_third_df.csv',
        help="input csv with video names")
    parser.add_argument("--vids_dir", type=str, default='/data/bball/edge-100/full_clips_Kent_0915/',
        help='location for video files')
    parser.add_argument("--output", type=str, default='/data/bball/edge-100/csv/09152020_third_df_ReleaseFrame.csv',
        help="path to output file (as CSV)")
    # Sensitive to lighting. Set ~50. for well-lit clips, at ~30. for moderate lighting, and as low as 15-20 for poorly lit.
    # TODO: Learn a parameter based on overal lit of the full frame? Should not be difficult.
    parser.add_argument("--min_circ_conf", type=float, default=30.,
        help='Confidence in circle (baseball) in/out. Set ~50 for good light, as low as 15-20 for bad lighting...')
    parser.add_argument("--debug", action="store_true", default=False,
        help='extensive debug, including saving frames')
    parser.add_argument("--debug_dir", type=str, default='/data/bball/edge-100/scratch/Kent_0915/',
        help="where to save frames and boxes")
    args = parser.parse_args()
    print(args)

    debug = args.debug

    # Source videos
    df = pd.read_csv(args.input)

    # Massive hack -- restrict to short list of videos
    if len(vids_interest) > 0:
        df['name_canon'] = df['VideoFile'].apply(lambda x: pathlib.Path(x).stem)
        df = df[df['name_canon'].isin(vids_interest)]
        df = df.reset_index(drop=True)

    df = df.drop(columns=['y1_new','x2_new','y2_new','x1_new','ReleaseFrame'], errors='ignore')
    #df = df.head(10)
    if sum(df['VideoFile'].notnull()) == 0:
        print('No vids found')
        return
    matched_videos = list(df.VideoFile)
    VIDEOFOLDER_PATH = args.vids_dir

    ReleaseFrame,x1,y2,x2,y1,h,w = [],[],[],[],[],[],[]
    found_frame,no_frame = 0,0
    for vid in tqdm.tqdm(matched_videos, total=len(matched_videos)):
        if debug:
            print('Processing video...')
            print(vid)

        coords = release_frame(VIDEOFOLDER_PATH+'/'+vid, debug=debug, args=args)

        if debug:
            print('Found release frame coordinates')
            print(coords)

        #continue
        #assert False

        if pd.isna(coords):
            no_frame += 1
            print('not found ReleaseFream for vid: ' + vid)
            print('Found ReleaseFrame for %d/%d vids' % (found_frame, (found_frame+no_frame)))
            ReleaseFrame.append(None)
            x1.append(None)
            y2.append(None)
            x2.append(None)
            y1.append(None)
            w.append(None)
            h.append(None)
        else:
            found_frame += 1
            if debug:
                print('Found coordinates ', vid)
            ReleaseFrame.append(coords[0])
            x1.append(coords[1])
            y2.append(coords[2])
            x2.append(coords[3])
            y1.append(coords[4])
            w.append(coords[5])
            h.append(coords[6])

    #assert False
    print('Found ReleaseFrame for %d/%d vids' % (found_frame, (found_frame+no_frame)))

    df['No'] = df.index + 1
    df.insert(2, 'ReleaseFrame', df.apply(lambda row: ReleaseFrame[int(row['No'])-1], axis=1))
    df.insert(2, 'x1_new', df.apply(lambda row: x1[int(row['No'])-1], axis=1))
    df.insert(2, 'y2_new', df.apply(lambda row: y2[int(row['No'])-1], axis=1))
    df.insert(2, 'x2_new', df.apply(lambda row: x2[int(row['No'])-1], axis=1))
    df.insert(2, 'y1_new', df.apply(lambda row: y1[int(row['No'])-1], axis=1))
    df.insert(2, 'frame_height', df.apply(lambda row: h[int(row['No'])-1], axis=1))
    df.insert(2, 'frame_width', df.apply(lambda row: w[int(row['No'])-1], axis=1))
    print('Saving updated data '+str(df.shape)+' '+args.output)
    df.to_csv(args.output)

if __name__ == "__main__":
    main()