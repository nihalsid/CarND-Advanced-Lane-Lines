## Writeup 


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

[image1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/color_binary.jpg "Binary Example"
[image4]: ./output_images/perpective_test.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image7]: ./camera_cal/calibration1.jpg "Calibration"
[image8]: ./test_images/perspective.jpg "Warp Example-1"
[image9]: ./test_images/straight_lines1.jpg "Warp Example-2"
[image10]: ./output_images/perspective_test2.jpg "Warp Example-3"
[image11]: ./output_images/test1_dist_corrected.jpg "Distortion corrected"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in function `camera_calibration()` of the the file `submission.py` (line 12 to 36).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imagepoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imagepoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image7]

is corrected to 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Example: 
![alt text][image2]
is corrected to 
![alt text][image11]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 44 through 63 in `submission.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp_image()`, which appears in lines 66 through 74 in the file `submission.py`.  The `unwarp_image()` function takes as inputs an image (`image`). I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[600, 447], [680, 447], [270, 673], [1037, 673]])
dst = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 447     | 300, 0        | 
| 680, 447     | 950, 0      |
| 270, 673     | 300, 720      |
| 1037, 673      | 950, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

is warped to

![alt text][image4]

Also,

![alt text][image9]

is warped to

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In function `get_annotated_frame()` in `submission.py`, line 87 to 139, I use a histogram to get the start of the two lane lines by identifying peaks in the histogram. Then I use a sliding window moving upwards to identify where the lane lines lie. After collecting the points for left and right lane line via sliding window, I fit polynomials on both of them using `np.polyfit()` line 135 :

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 141 through 153 in my code in `submission.py`. For this I used the curvature formula discussed in the lecture and the pixel to meter conversions also discussed in the lectures.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 164 through 169 in my code in `submission.py` in the function `get_annotated_frame()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The thresholds for binarization are still not robust. Needs more tuning to make it more robust. Also, I do the whole sliding window search on each frame, which can be avoided by just using information from the last frame's estimated line. 
