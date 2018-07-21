# Vehicle Detection

This project contains code and media related to the "Vehicle Detection" project for Udacity Self Driving Car NanoDegree.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Contents:

* /output_video - video of vehicle detection result for "project_video" and "test_video"
* /output_images - contains images used in writeup
* /test_images - images provided for pipeline testing for project
* Project_Notebook.ipnyb - Jupyter notebook that contains steps taken towards completion of the project including the pipeline
* lanedetection.py - pipeline for Lane Detection
* project_writeup.md - contains project pipeline summary and reflections on the project outcome
* save.pkl - pickle file containing SVM classifier and feature extraction parameter values
* var.pkl - pickle file containing camera calibration calculation for Lane Detection pipeline
