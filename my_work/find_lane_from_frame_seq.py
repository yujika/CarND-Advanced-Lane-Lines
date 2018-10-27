# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:32:22 2018

@author: yujika
"""
import pickle
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util
from moviepy.editor import VideoFileClip
import traceback

ksize = 9 # Choose a larger odd number to smooth gradient measurements
##########################
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

#FRAME REFERENCE - how many frames to be taken into account?
ref_frames = 10

def process_image(image_org):
    global window_pos
    try:
        # picked up image processing from find_lane.py after it turned out working with still image.
        image_undistort = cv2.undistort(image_org, mtx, dist, None, mtx)
        image_normalize = (image_undistort - np.mean(image_undistort))/np.std(image_undistort)*32+128
        R = image_normalize[:,:,0]
        G = image_normalize[:,:,1]

        hls_binary = util.hls_select(image_undistort, thresh=(20, 255))
        plt.title('S channel with thresh')
        plt.imshow(hls_binary, cmap='gray')
        plt.show()
        
        gradx = util.abs_sobel_thresh(hls_binary, orient='x', sobel_kernel=ksize, thresh=(40, 100))
        grady = util.abs_sobel_thresh(hls_binary, orient='y', sobel_kernel=ksize, thresh=(40, 100))
        grad = np.zeros_like(hls_binary)
        grad[((gradx == 1) & (grady == 1) ) ] = 1
        plt.title('grad')
        plt.imshow(grad, cmap='gray')
        plt.show()
        
        
        mag_binary = util.mag_thresh(hls_binary, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = util.dir_threshold(hls_binary, sobel_kernel=ksize, thresh=(-0.6*(np.pi/2), 0.6*(np.pi/2)))
        combined = np.zeros_like(hls_binary)
        combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | ((R>128+32) & (G>128+32))] = 1
        bird_view = util.warper(combined, src_points, dst_points)
        
        if( len(history_left_x) > 0 ):
            if ( len(history_left_x) < ref_frames ):
                end_pos = -1 - len(history_left_x)
            else:
                end_pos = -1 - ref_frames
            leftx_with_hist = np.hstack(history_left_x[-1:end_pos:-1]).flatten() 
            lefty_with_hist = np.hstack(history_left_y[-1:end_pos:-1]).flatten()
            rightx_with_hist = np.hstack(history_right_x[-1:end_pos:-1]).flatten()
            righty_with_hist = np.hstack(history_right_y[-1:end_pos:-1]).flatten()
        else:
            leftx_with_hist = np.empty([0,])
            lefty_with_hist = np.empty([0,])
            rightx_with_hist = np.empty([0,])
            righty_with_hist = np.empty([0,])
        
        leftx, lefty, rightx, righty, window_pos_ret = util.find_lane_pixels_with_history(bird_view, nwindows, margin, minpix, window_pos, leftx_with_hist, lefty_with_hist, rightx_with_hist, righty_with_hist)
        window_pos = window_pos_ret[0]
        # save for future reuse
        history_left_x.append( [leftx] )
        history_left_y.append( [lefty] )
        history_right_x.append( [rightx] )
        history_right_y.append( [righty] )
#        color_fit_lines = np.zeros( bird_view.shape + ( 3, ), dtype=np.int32 )
#        color_fit_lines[lefty,leftx] = [255, 0, 0]
#        color_fit_lines[righty,rightx] = [0, 0, 255 ]
        
        if( len(history_left_x) > 1 ):
            if ( len(history_left_x) < ref_frames ):
                end_pos = -1 - len(history_left_x)
            else:
                end_pos = -1 -ref_frames
            leftx_with_hist = np.hstack(history_left_x[-1:end_pos:-1]).flatten() 
            lefty_with_hist = np.hstack(history_left_y[-1:end_pos:-1]).flatten()
            rightx_with_hist = np.hstack(history_right_x[-1:end_pos:-1]).flatten()
            righty_with_hist = np.hstack(history_right_y[-1:end_pos:-1]).flatten()
        else:
            leftx_with_hist = leftx
            lefty_with_hist = lefty
            rightx_with_hist = rightx
            righty_with_hist = righty
            
#        color_fit_lines[lefty_with_hist,leftx_with_hist] = [255, 0, 0]
#        color_fit_lines[righty_with_hist,rightx_with_hist] = [0, 0, 255 ]
            
        pic_height = bird_view.shape[0]
        left_fitx, right_fitx = util.fit_polynomial(leftx_with_hist, lefty_with_hist, rightx_with_hist, righty_with_hist, pic_height )
        ploty = np.linspace(0, pic_height-1, pic_height )
        color_with_lane_region = np.zeros( bird_view.shape + ( 3, ), dtype=np.uint8 )
        poly = np.vstack( ( np.array([left_fitx,ploty],dtype=np.int32).T , np.array([right_fitx,ploty],dtype=np.int32).T[::-1] ) )
        color_with_lane_region = cv2.fillPoly( color_with_lane_region, [poly] , color=[0,32,0])
        color_with_lane_region[lefty_with_hist,leftx_with_hist] = [255, 0, 0]
        color_with_lane_region[righty_with_hist,rightx_with_hist] = [0, 0, 255 ]
        window_height = np.int(color_with_lane_region.shape[0]//nwindows)
        for ( l, r, h ) in window_pos_ret:
            cv2.rectangle(color_with_lane_region, (l-margin, h - window_height), ( l+ margin, h ), ( 255,0,0 ),3 )
            cv2.rectangle(color_with_lane_region, (r-margin, h - window_height), ( r+ margin, h ), ( 0,0,255 ),3 )
        cv2.imshow('line', color_with_lane_region)
        cv2.waitKey(1) 
        left_fit_in = left_fitx[::-1]  # Reverse to match top-to-bottom in y
        right_fit_in = right_fitx[::-1]  # Reverse to match top-to-bottom in y
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit_in*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit_in*xm_per_pix, 2)
        left_curverad, right_curverad = util.measure_curvature_real(left_fit_cr, right_fit_cr, ploty, xm_per_pix, ym_per_pix )
        curverad = np.mean( [left_curverad, right_curverad])
        center = left_fit_in[0] + (right_fit_in[0] - left_fit_in[0])/2
        camera_center_pix = bird_view.shape[1]/2
        if ( center == camera_center_pix):
            car_position_string = "CENTER"
        elif ( center < camera_center_pix):
            diff = (camera_center_pix - center)*xm_per_pix
            car_position_string = "Vehcle is {:.2f}m left of center".format(diff)
        else:
            diff = (center - camera_center_pix)*xm_per_pix
            car_position_string = "Vehcle is {:.2f}m right of center".format(diff)
        camera_view = util.warper(color_with_lane_region, dst_points, src_points )
        img_overlayed = util.weighted_img(camera_view.astype(np.uint8), image_undistort )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_start = int(img_overlayed.shape[1]/100)
        y_start = int(img_overlayed.shape[0]/10)
        y_inc = int(img_overlayed.shape[0]/20)
        font_size = 1.0
        cv2.putText(img_overlayed,'R={:>8d} meter'.format(int(curverad)),(x_start,y_start), font, font_size, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img_overlayed,car_position_string,(x_start,y_start+y_inc*1), font, font_size, (255,255,255), 2, cv2.LINE_AA)
    except TypeError:
        traceback.print_exc()
        img_overlayed = image_org

    cv2.imshow('debug',img_overlayed)
    cv2.waitKey(1)        
    return img_overlayed



dist_pickle = pickle.load( open( util.camera_mtx_file_name, "rb" ) )
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
src_points = dist_pickle['src_points']
dst_points = dist_pickle['dst_points']

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

cv2.namedWindow('debug',cv2.WINDOW_NORMAL)
cv2.namedWindow('line',cv2.WINDOW_NORMAL)


#########For video test
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
if ( 1 ):
    history_left_x = []
    history_left_y = []
    history_right_x = []
    history_right_y = []
    window_pos = (0,0)
    clip1 = VideoFileClip("../project_video.mp4")
    project_video = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    project_video.write_videofile('../output_videos/project_video_processed.mp4', audio=False)

if ( 1 ):
    history_left_x = []
    history_left_y = []
    history_right_x = []
    history_right_y = []
    window_pos = (0,0)
    clip2 = VideoFileClip("../challenge_video.mp4")#.subclip(7,8)
    challenge_video = clip2.fl_image(process_image)
    challenge_video.write_videofile('../output_videos/challenge_video_processed.mp4', audio=False)

if ( 1 ):    
    history_left_x = []
    history_left_y = []
    history_right_x = []
    history_right_y = []
    window_pos = (0,0)
    clip3 = VideoFileClip("../harder_challenge_video.mp4")
    harder_challenge_video = clip3.fl_image(process_image)
    harder_challenge_video.write_videofile('../output_videos/harder_challenge_video_processed.mp4', audio=False)
    
    
    cv2.destroyAllWindows()
#FIN
