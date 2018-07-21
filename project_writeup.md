## Project Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_Features.png
[image2]: ./output_images/sliding_window.png
[image3]: ./output_images/find_cars.png
[image4]: ./output_images/heat_map.png
[image5]: ./output_images/bounding_box.png
[video1]: ./output_video/project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
You can find the implementation of HOG feature extraction under `HOG Features` of the `Project_Notebook.ipynb`

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the `vehicle` and `non-vehicle` datasets in order to find the optimal parameter and color space combination .

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and in my opinion `U` channel of the `LUV` color space the most distinct features across vehicle and non-vehicle images due to its invariance to light coditions.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

SVM is trained under `Train SVM Classifier` section of the `Project_Notebook.ipynb`. I used `LinearSVC`with `hinge` loss function. Below are the parameters used for extracting features for training: 

* pix_per_cell = 8 - HOG parameter
* cell_per_block = 3 - HOG parameter
* orient = 12 - HOG parameter
* cspace='LUV' - color space for Color Histogram and HOG extractions
* hog_channel = 1 - Sets 'U' channel of 'LUV' color space for HOG extraction
* spatial_size=(32, 32) - Image dimensions used for spatial features


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I experimented with different window sizes and overlap ratios in order to find the optimal setting. Below are the final parameters used:

* window size (128x128 px) 
* overlap (0.85, 0.85)
* y_start_stop = (350,650)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the same values in feature extraction that I used for extracting features of training data. Here is an example output:

![alt text][image2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used `find_cars` function to understand the performance of HOG subsampling at different steps, scales and areas. Here is an example using ystart = 400, ystop = 656, scale = 1.5, step = 2, threshold = 1:

![alt text][image2]

After these investigations I decided to scan the image with lower scaling and step values at the horizon while increasin these values as it approached closer to the camera.

My pipeline starts with `id_cars` function which takes a list of scales and corresponding areas to search for cars in an image. Values in this list should be ordered from so that smallest scale and area furthest from the camera should be the first element, and following values should increase in scale while moving towards the camera. The `step` parameter is hard coded to initialize with one and increment by one  for each (scale, area) pair in the lists.

These values are then submitted to `find_cars` function to find the cars in the image by using HOG sampling and retrieve bounding boxes for these positive identifications. These box coordeinates are then submitted to `add_heat` function which creates a map of pixel scores (heatmap) according to how many times a pixel appears in an region identified as a car.`apply_threshold` function takes in the heatmap and filters out the pixels that have a score below the given threshold. Finally, using `scipy.ndimage.measurements.label()` on the heatmap process extracts the regions identified as cars and draws overlays a bounding box on the video image.

`apply_threshold` function combined with multiple runs at different scales and steps over multiple regions helps filtering the false postive identifications.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![alt text][image4]

### Here is an example of bounding boxes are drawn onto the last frame in the series:
![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The biggest problem was to strike a balance to filter out false positive identification of non-vehicle objects while keeping the positive identification of vehicles on the road. I had to experiment with many different scale, step and region combinations before I could settle on a solution that was satisfactory.

* The pipeline is likely to fail where images are far too different to those in the training dataset such as differences in light conditions, vehicle orientation etc. Furthermore, the pipeline might face issues in identifying vehicles that are moving faster through the frames, as perhaps glimpsed through its handling of oncoming traffic. 

* Pipeline can be improved by tracking positively identified vehicles as distinct objects through the frames. This would allow collection of series of identification regions over multiple frames for a specific car which then can help in improvements such as: eliminating false positives, smoothing bounding boxes across frames, avoiding box merging when two vehicles overlap each other.

* Finally, a deep NN classifier would probably provide improvements upon the SVM model used in the project when dealing with rapidly changing conditoins across frames.
 

