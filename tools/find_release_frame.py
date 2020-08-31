import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import re
import pandas as pd
import glob
import pathlib

def is_baseball(img, frame=0, cnt=0, scratch_dir='./', fname=''):
    # dimensions
    h = img.shape[0]
    w = img.shape[1]

    print('looking for baseball -- hxw '+str([h,w]))

    # convert to grayscale, and then two different binary treshholds
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, thresh = cv.threshold(gray, 0, 255, 10)
    #ret, trunc = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    # detect circles
    print('looking for a circle...')
    min_rad = int(max(h,w)/3)
    min_dist = int(max(h,w)/6)
    print('Looking for circle with min radius, min_dist: '+str([min_rad, min_dist]))
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=50, param2=30, minRadius=min_rad, maxRadius=int(w))

    print(circles)

    if not(circles is not None):
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
    print('Drawing circles to file: '+cnt_file)
    cv.imwrite(cnt_file, frame_w_circles)
    cv.imwrite(green_cnt_file, green)

    circles = np.round(circles[0, :]).astype("int")
    # should only be one circle, otherwise probably not a baseball
    if len(circles) > 2 or green_ratio > max_full_image_green_ratio:
        return False

    # make sure something was detected -- contrast between inside and outside the circle
    min_conf = 10. # 25. # 60.
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
        slack = max(h,w) * 0.05
        in_bounds = (x - r > 0 - slack) and (x + r < w + slack) and (y - r > 0 - slack) and (y + r < h + slack)

        # Make sure the ball is not green/green screen (probably not a ball)
        green_ball_sum = np.sum(imask * ball_mask)
        all_ball_sum = np.sum(ball_mask)
        green_ball_ratio = green_ball_sum / all_ball_sum

        print('conf, in_bounds, green_ball_ratio'+str([conf, in_bounds, green_ball_ratio]))
        if conf >= min_conf and in_bounds and green_ball_ratio < max_green_ratio:
            return (x, y, r)

    print('No circles found')
    return False

'''
Takes in path to video and returns the frame # at ball-release if detected, otherwise None
'''
def release_frame(vid_path, min_frame=100, max_frame=700, scratch_dir = '/data/bball/edge-100/scratch/'):
    print('--------------\nNew file: ', vid_path)
    vid = cv.VideoCapture(vid_path)
    fname = pathlib.Path(vid_path).stem

    # use for motion
    subtract = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    i = 0
    while i < max_frame:
        isFrame, frame = vid.read()
        if not isFrame: break

        if i > min_frame:
            #image_file = scratch_dir + 'frame_'+str(i)+'_'+fname+'.jpg'
            image_file = scratch_dir + 'frame_'+str(i)+'.jpg'
            cv.imwrite(image_file, frame)

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

        if len(contours) > 0:
            print('Frame '+str(i)+' found contours '+str(len(contours)))
        #print(contours)

        for ci, cnt in enumerate(contours):
            # must be the expected size of a baseball
            a = cv.contourArea(cnt)
            if a in range(1000,10000): # range(1000,2000): #range(4000,10000):
                x,y,w,h = cv.boundingRect(cnt)

                print('Examining contour (x,y,w,h):')
                print([x,y,w,h])
                print('area '+str(a))

                # must be away from edge or the image, square-like, and fill up most of bounding box
                border = 25
                in_middle = y > border and y < frame_h - border and x > border and x < frame_w - border
                square = abs(1-(min(w,h)/max(w,h))) < 0.15 # 0.075
                print('squariness '+str(abs(1-w/h)))
                filled = a/(w*h) > 0.75
                print('filledness '+str(a/(w*h)))
                # if meets all requirements, release-frame is detected and returned
                print('in_middle, square, filled: '+str([in_middle, square, filled]))
                print('attempting is_baseball on this item')

                if i > min_frame:
                    cnt_file = scratch_dir + 'frame_'+str(i)+'_cnt_'+str(ci)+'_'+fname+'.jpg'
                    cv.imwrite(cnt_file, frame[y:y+h, x:x+w])

                if in_middle and square and is_baseball(frame[y:y+h, x:x+w].copy(),frame=i,cnt=ci,scratch_dir=scratch_dir,fname=fname):
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    #plt.imshow(frame)
                    #plt.show()
                    print(i, x, y, x + w, y + h )

                    # Save bounding box super-imposed on the original image
                    image_file = scratch_dir + 'frame_'+str(i)+'_bbox_'+fname+'.jpg'
                    cv.imwrite(image_file, frame)

                    print('ReleaseFrame ('+str(i)+') found in vid!! '+str(vid_path))

                    return i, x, y, x + w, y + h
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

#matched_videos = ['Lange_A_PitchDesign_KentFacility_20-Aug-20_1597956120.mov', ]
vids = glob.glob('/data/bball/edge-100/full_clips_Kent/*mov')
matched_videos = [pathlib.Path(f).name for f in vids]


VIDEOFOLDER_PATH = '/data/bball/edge-100/full_clips_Kent'

ReleaseFrame = []; x1 = []; y2 = []; x2 = []; y1 = []
for vid in matched_videos:
    print('Processing video...')
    print(vid)

    coords = release_frame(VIDEOFOLDER_PATH + "/" + vid)

    print('Found release frame coordinates')
    print(coords)

    continue
    assert False

    if pd.isna(coords):
        print('YO', vid)
    #if pd.isna(vid):
    #if vid == None:
        ReleaseFrame.append(None)
        x1.append(None)
        y2.append(None)
        x2.append(None)
        y1.append(None)
    else:
        print(vid)
        #release_frames.append(frame)
#             ReleaseFrame.append(coords)
        ReleaseFrame.append(coords[0])
        x1.append(coords[1])
        y2.append(coords[2])
        x2.append(coords[3])
        y1.append(coords[4])

assert False

df['No'] = df.index + 1
df.insert(2, 'ReleaseFrame', df.apply(lambda row: ReleaseFrame[int(row['No'])-1], axis=1))
df.insert(2, 'x1_new', df.apply(lambda row: x1[int(row['No'])-1], axis=1))
df.insert(2, 'y2_new', df.apply(lambda row: y2[int(row['No'])-1], axis=1))
df.insert(2, 'x2_new', df.apply(lambda row: x2[int(row['No'])-1], axis=1))
df.insert(2, 'y1_new', df.apply(lambda row: y1[int(row['No'])-1], axis=1))
df.to_csv(file_output)