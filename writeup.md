**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_straight_lines1.png "Road Transformed"
[image3]: ./output_images/binary_combo_test6.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines_test1.jpg "Fit Visual"
[image6]: ./output_images/output_test1.jpg "Output"
[video1]: ./output_videos/project_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1.  the camera matrix and distortion coefficients. 
I used Camera calibration code from Lesson 5 Camera Calibration and implemented in `my_work/camera_calibration.py`.

Provided checker board images are processed by findChessboardCorners().
Iterate through all the checker board images.
All the 3d points `objpints` and 2d points `imgpoints` are passed to `cv2.calibrateCamera()` to get camera matrix. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Camera matrix is picled at line 152 of `my_work/camera_calibration.py` and re-used in `my_work/find_lane.py` and `my_work/find_lane_from_frame_seq`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 35 through 57 in `find_lane_from_frame_seq.py`).
R and G channel for line color detection.
Gradient, mag and dir thresholds are applied to S channel.   
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `util.warper()`, which appears in lines 49 through 56 in the file `util.py` (my_work/util.py).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
```python
    src_points = np.float32([[571,467],
                             [717,467],
                             [1105,720],
                             [205,720]])
    dst_points = np.float32([[1280/4,   0 ],
                             [1280/4*3, 0 ],
                             [1280/4*3, 720],
                             [1280/4,   720]
                             ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 571, 467      | 320, 0        | 
| 717, 467      | 960, 0        |
| 1105, 720     | 960, 720      |
| 205, 460      | 320, 720      |

Those points are pickled at line 152 of `my_work/camera_calibration.py`.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identifying lane-line pixels and fit their positions with a polynomial

- get HLS image
- get R and G channel of a normalized image, which help detect white line ( S channel is not so good at detecting white line ).
- combine multiple image processing result and get combined binary image
- get mean of left and right line ROI and set it as a center window position.
- set +- 100 pixels around the center position are sliding window.
- devide image size by 9 for window height
- in case for processing sequential images.
    - window position is saved and re-used for finding next frame's lane line position.
    - binary output from previous frames are also re-used to detect lane line position, which especially useful when lane line is not contiguous.

![alt text][image5]

#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

- for the radius calculation
  line 318 through 330 in my code in `my_work/util.py`
- for statis image center calculation
  lines 184 through 194 in my code in `my_work/find_lane.py`
- for sequential images center calculation
  lines 124 through 125 in my code in `my_work/find_lane_from_frame_seq.py`

#### 6. Resulting image of the lane area.

I implemented this step in lines 196 through 198 in my code in `my_work/find_lane.py`.
```python
    #Warp the detected lane boundaries back onto the original image.
    camera_view = util.warper(color_with_lane_region, dst_points, src_points )
    img_overlayed = util.weighted_img(camera_view.astype(np.uint8), image_undistort )
```
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_processed.mp4)

Here's a [link to challenge video result](./output_videos/challenge_video_processed.mp4.mp4)
Here's a [link to harder challenge video result](./output_videos/harder_challenge_video_processed.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- approach
    - Utilize R and G channel to help increase weight of lane line
        - R and G channel includes luminance, so I normalized scene luminance as below.
```python
        image_normalize = (image_undistort - np.mean(image_undistort))/np.std(image_undistort)*32+128
```
( line 36 of find_lane_from_frame_seq.py )
    - Utilize previous frame's binary combo result
        - Add past binary combo result in side window
        - Add those from past several frames
    - Utilize center line distance to identify possible center line position

- issues
    - S channel can't detect dotted lane line well
    - R and G channel tend to pick up unwanted object, like a white car, or noisy
    - R and G channel unreliable if roard is kind of white color.
    - Dotted line is so sparse, can't fit polyline well
    - Side wall is misunderstood as a lane line
    - Scene luminance change cause false lane detection a lot
    - Tree shadow causes false lane detection
    - Road crack causes false lane detection
    - Winding road can't be detected well


- My current pipeline
    - R and G channel pick up false point a lot if road is kind of white color.
        - Color processing need to be enhanced more.
            - Just cancel R and G channel if too many points are detected.
            - Do R and G channel detection only within window area to normalize image within window area.
    - Fails with curved line such as in `harder_challenge_video.mp4`
        - Wider ROI region - but this may cuase false detection of other scenes more.
        - Probably need to detect almost horizontal lane line or a lane line completely dissapeared from a seen. Windowed approach should be improved.
        - My current pipeline shows lane line well continuous from several frames, so line tracing may work
    - Sometimes the pipeline detect the self car body as a starting point of lane line
        - Exclude know object from ROI

