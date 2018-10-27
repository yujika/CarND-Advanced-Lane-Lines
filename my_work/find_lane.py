# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:57:10 2018

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

dist_pickle = pickle.load( open( util.camera_mtx_file_name, "rb" ) )
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
src_points = dist_pickle['src_points']
dst_points = dist_pickle['dst_points']

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

test_file_name = glob.glob('../test_images/test*.jpg')
for image_file in test_file_name:
    image_org = mpimg.imread(image_file)
#    plt.imshow(image)
#    plt.show()
    
    #undistort
    image_undistort = cv2.undistort(image_org, mtx, dist, None, mtx)
    plt.imshow(image_undistort)
    plt.show()
    
    #plt.imshow( image - image_undistort )
    
    image = image_undistort
    
    #### Just for experimental
#    thresh = (180, 255)
#    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#    binary = np.zeros_like(gray)
#    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
#    
#    R = image[:,:,0]
#    G = image[:,:,1]
#    B = image[:,:,2]
#    
#    
#    thresh = (200, 255)
#    binary = np.zeros_like(R)
#    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
#    
#    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#    H = hls[:,:,0]
#    L = hls[:,:,1]
#    S = hls[:,:,2]
    
#    plt.title('H')
#    plt.imshow(H,cmap='gray')
#    plt.show()
#    plt.title('L')
#    plt.imshow(L,cmap='gray')
#    plt.show()
#    plt.title('S')
#    plt.imshow(S,cmap='gray')
#    plt.show()
    
    
    hls_binary = util.hls_select(image_undistort, thresh=(90, 255))
 #   plt.title('S channel with thresh')
 #   plt.imshow(hls_binary, cmap='gray')
 #   plt.show()
    
#    thresh = (90, 255)
#    binary = np.zeros_like(S)
#    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
#    
#    thresh = (15, 100)
#    binary = np.zeros_like(H)
#    binary[(H > thresh[0]) & (H <= thresh[1])] = 1
#    
    
 #   plt.title('original image')
 #   plt.imshow(image)
 #   plt.show()
    
    # Is it better to normalize?
     
    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = util.abs_sobel_thresh(hls_binary, orient='x', sobel_kernel=ksize, thresh=(40, 100))
 #   plt.title('gradx')
 #   plt.imshow(gradx,cmap='gray')
 #   plt.show()
    grady = util.abs_sobel_thresh(hls_binary, orient='y', sobel_kernel=ksize, thresh=(40, 100))
 #   plt.title('grady')
 #   plt.imshow(grady,cmap='gray')
 #   plt.show()
    mag_binary = util.mag_thresh(hls_binary, sobel_kernel=ksize, mag_thresh=(30, 100))
 #   plt.title('magnitude')
 #   plt.imshow(mag_binary,cmap='gray')
 #   plt.show()
    dir_binary = util.dir_threshold(hls_binary, sobel_kernel=ksize, thresh=(0.7, 1.3))
 #   plt.title('direction')
 #   plt.imshow(dir_binary,cmap='gray')
 #   plt.show()
    combined = np.zeros_like(hls_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
 #   plt.title('COMBINED')
 #   plt.imshow(combined,cmap='gray')
 #   plt.show()
#    plt.imsave('combined_gray_scale.png',combined) # To check src points manually.
    
    
    base_fn = os.path.basename(image_file).split('.')[0]
    plt.imsave('../output_images/' + 'binary_combo_' + base_fn + '.jpg', combined*255, cmap='gray' )
        
        
    bird_view = util.warper(combined, src_points, dst_points)
    plt.title('COMBINED bird view ' + image_file)
    plt.imshow(bird_view,cmap='gray')
    plt.show()
    plt.imsave('../output_images/' + 'binary_combo_warped_' + base_fn + '.jpg', bird_view*255, cmap='gray' )

    ##########################
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    leftx, lefty, rightx, righty = util.find_lane_pixels(bird_view, nwindows, margin, minpix)
    
    color_fit_lines = np.zeros( bird_view.shape + ( 3, ), dtype=np.int32 )
    color_fit_lines[lefty,leftx] = [255, 0, 0]
    color_fit_lines[righty,rightx] = [0, 0, 255 ]
    
    pic_height = bird_view.shape[0]
    left_fitx, right_fitx = util.fit_polynomial(leftx, lefty, rightx, righty, pic_height )
    ploty = np.linspace(0, pic_height-1, pic_height )
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='green')
    plt.imshow(color_fit_lines)
    plt.savefig('../output_images/' + 'color_fit_lines_' + base_fn + '.jpg')
    plt.show()
    
    color_with_lane_region = np.zeros( bird_view.shape + ( 3, ), dtype=np.int32 )
    poly = np.vstack( ( np.array([left_fitx,ploty],dtype=np.int32).T , np.array([right_fitx,ploty],dtype=np.int32).T[::-1] ) )
    color_with_lane_region = cv2.fillPoly( color_with_lane_region, [poly] , color=[0,32,0])
    color_with_lane_region[lefty,leftx] = [255, 0, 0]
    color_with_lane_region[righty,rightx] = [0, 0, 255 ]
    plt.imshow(color_with_lane_region)
    plt.savefig('../output_images/' + 'color_with_lane_region_' + base_fn + '.jpg')
    plt.show()
    

    left_fit_in = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    right_fit_in = right_fitx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit_in*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit_in*xm_per_pix, 2)

    #left_curverad, right_curverad = util.measure_curvature_pixels(left_fitx, right_fitx, ploty )
    left_curverad, right_curverad = util.measure_curvature_real(left_fit_cr, right_fit_cr, ploty, xm_per_pix, ym_per_pix )
    print(left_curverad, right_curverad )
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
    print(car_position_string)
    
    #Warp the detected lane boundaries back onto the original image.
    camera_view = util.warper(color_with_lane_region, dst_points, src_points )
    img_overlayed = util.weighted_img(camera_view.astype(np.uint8), image_undistort )
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_start = int(img_overlayed.shape[1]/100)
    y_start = int(img_overlayed.shape[0]/10)
    y_inc = int(img_overlayed.shape[0]/20)
    font_size = 1.0
    cv2.putText(img_overlayed,'R={:} meter'.format(curverad),(x_start,y_start), font, font_size, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img_overlayed,car_position_string,(x_start,y_start+y_inc*1), font, font_size, (255,255,255), 2, cv2.LINE_AA)
    
    
    plt.imshow(img_overlayed)
    plt.show()
    plt.imsave('../output_images/' + 'output_' + base_fn + '.jpg', img_overlayed )

    


