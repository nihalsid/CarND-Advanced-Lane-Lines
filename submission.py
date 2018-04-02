import cv2
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

USE_SAVED_CALIB = True


def camera_calibration(path_to_chessboards):
	# camera calibration by using chessboards
	imagepoints = []
	objpoints = []
	image_size = None
	# obj points lie on flat plane with z = 0
	objp = np.zeros((9*6, 3), np.float32)
	objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2);
	
	images = glob.glob(path_to_chessboards)
	
	for fname in images:
		image = mpimg.imread(fname)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image_size = gray.shape[::-1]
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		# add to collection of imagepoints and objpoints if pattern found
		if ret == True:
			imagepoints.append(corners)
			objpoints.append(objp)
			image_with_corners = cv2.drawChessboardCorners(image, (9,6), corners, ret)
			cv2.imwrite(os.path.join('save_files', os.path.basename(fname)), image_with_corners)
	# return the calib matrix and dist correction coefficients
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, image_size, None, None)
	return mtx, dist


def undistort(image, mtx, dist):
	# undistort an image using matrix and coefficients of distortion
	return cv2.undistort(image, mtx, dist, None, mtx)


def create_binary_image(image, s_thresh=(180, 255), sx_thresh=(15, 100)):
	# creates binary image based on thresholds on s channel and the sobel derivative of l channel
	img = np.copy(image)
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# derivative
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx) 
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	# s channel thresholding 
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
	binary = np.zeros_like(s_channel)
	# concatenate
	binary[(sxbinary != 0) | (s_binary!=0)] = 255
	return binary


def unwarp_image(image):
	# do perspective transformation
	src = np.float32([[600, 447], [680, 447], [270, 673], [1037, 673]])
	dst = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])
	# 640 gets mapped to 625 - which will be used in distance from center calculation
	img_size = (image.shape[1], image.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	unwarped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_NEAREST)
	return unwarped


def warp_image(image):
	# reverse of unwarp image
	dst = np.float32([[600, 447], [680, 447], [270, 673], [1037, 673]])
	src = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])
	img_size = (image.shape[1], image.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_NEAREST)
	return warped


def get_annotated_frame(original_frame, thresholded_frame):
	# this function annotates a frame with lane region, curvature and distance from center

	# use histogram peaks to find the lane lines starting points
	histogram = np.sum(thresholded_frame[thresholded_frame.shape[0]//2:,:], axis=0)
	midpoint = np.int(histogram.shape[0]//2)

	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# use sliding windows to follow the lane lines
	nwindows = 9
	window_height = np.int(thresholded_frame.shape[0]//nwindows)
	nonzero = thresholded_frame.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	margin = 100
	minpix = 50
	left_lane_inds = []
	right_lane_inds = []
	# find left lane indices and right lane indices using sliding windows
	for window in range(nwindows):
		win_y_low = thresholded_frame.shape[0] - (window+1)*window_height
		win_y_high = thresholded_frame.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	ploty = np.linspace(0, thresholded_frame.shape[0]-1, thresholded_frame.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Curvature calculation
	y_eval = np.max(ploty)
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	dist_from_center = abs((left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2] + right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])/2 - 625)
	dist_from_center = dist_from_center * xm_per_pix
	curvature = (left_curverad + right_curverad) / 2
	print(curvature, 'm', dist_from_center, 'm')

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(thresholded_frame).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = warp_image(color_warp)
	# Combine the result with the original image
	result = cv2.addWeighted(original_frame, 1, newwarp, 0.3, 0)
	cv2.putText(result,'Radius of Curvature: %.2fm'%curvature,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(result,'Distance from center: %.2fm'%dist_from_center,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	return result
		

if __name__=='__main__':
	if not USE_SAVED_CALIB:
		mtx, dist = camera_calibration("camera_cal/*.jpg")
	else:
		mtx = pickle.load(open("saved_cal/mtx.p", "rb"))
		dist = pickle.load(open("saved_cal/dist.p", "rb"))
	
	# Define the codec and create VideoWriter object
	writer = cv2.VideoWriter('project_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280,720))
	vidcap = cv2.VideoCapture('project_video.mp4')
	ret, frame = vidcap.read()
	
	while ret:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out = undistort(frame, mtx, dist)
		out = create_binary_image(out)
		out = unwarp_image(out)
		out = get_annotated_frame(frame, out)
		out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
		writer.write(out)
		ret, frame = vidcap.read()
		
	vidcap.release()
	writer.release()
	