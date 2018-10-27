# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:51:28 2018

@author: yujika
"""

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt
import util




# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)
        plt.imshow(img)
        plt.show()

#cv2.destroyAllWindows()

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)




test_file_name = glob.glob('../test_images/straight_lines*.jpg')
for image_file in test_file_name:
    image_org = mpimg.imread(image_file)
    image_undistort = cv2.undistort(image_org, mtx, dist, None, mtx)
    plt.imshow(image_undistort)
    plt.show()
    image = image_undistort
    hls_binary = util.hls_select(image, thresh=(90, 255))
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    gradx = util.abs_sobel_thresh(hls_binary, orient='x', sobel_kernel=ksize, thresh=(40, 100))
    grady = util.abs_sobel_thresh(hls_binary, orient='y', sobel_kernel=ksize, thresh=(40, 100))
    mag_binary = util.mag_thresh(hls_binary, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = util.dir_threshold(hls_binary, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    
    src_points = np.float32([[571,467],
                             [717,467],
                             [1105,720],
                             [205,720]])
    dst_points = np.float32([[1280/4,   0 ],
                             [1280/4*3, 0 ],
                             [1280/4*3, 720],
                             [1280/4,   720]
                             ])
        
        
    bird_view = util.warper(combined, src_points, dst_points)
    plt.title('COMBINED bird view ' + image_file)
    plt.imshow(bird_view,cmap='gray')
    plt.show()
    base_fn = os.path.basename(image_file).split('.')[0]
    plt.imsave('../output_images/' + 'binary_combo_warped_' + base_fn + '.png', bird_view*255, cmap='gray'  )
    
    
    bird_view_rgb = util.warper(image_undistort, src_points, dst_points)
    bird_view_rgb = cv2.polylines(bird_view_rgb,np.array([dst_points],dtype=np.int32),True, ( 255, 0, 0) ,thickness=10)
    image_roi = cv2.polylines(image_undistort, np.array([src_points],dtype=np.int32),True, ( 255, 0, 0 ), thickness=10)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.tight_layout()
    ax1.imshow(image_roi)
    ax1.set_title('Original Image', fontsize=24)
    ax2.imshow(bird_view_rgb)
    ax2.set_title('bird view Image', fontsize=24)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('../output_images/' + 'warped_' + base_fn + '.jpg')
    plt.show()
    


save_pickle = {
        'mtx' : mtx,
        'dist': dist,
        'src_points' : src_points,
        'dst_points' : dst_points        
        }
with open(util.camera_mtx_file_name,'wb' ) as f:
    pickle.dump(save_pickle, f, pickle.HIGHEST_PROTOCOL)
    
