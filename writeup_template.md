#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

I opt to present a single image and it's label in the notebook. It's setup in a way a user can change the image by changing the number to switch what image is to display.

This is in cell 3 of the notebook

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Preprocessing steps start in cell 4 up to cell 8.

Images are turned into grayscale by using np.average. to turn the RGB channels into one dimension. (see cell 4-5)

Next is normalizing the "grayscaled" values close to one. with the understanding of images have a minimum of 0, and a maximum of 255 value. Grayscale values are normalized to a minimum value of 0.1, and a maximum value of 0.9. (see cell 6)

Since images have been grayscaled (changing the image shape from 32,32,3 to 32,32), it has to be reshaped into 32,32,1 in order for it to fit in the tensor. This is achieved by using np.reshape. (see cell 7)

Shuffle function is added in cell 8. This is to shuffle the order of a list of images and its corresponding labels.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data provided has already been split for training, validation, and testing. 

If data has to be split, this would be go about by first shuffling the data, then getting a fraction for each set using python's dictionary indexing.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Final model is found in cell 9. The model is almost exactly like the Lenet model which has the following layers below.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 			   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					| 	         									|
| Max pooling			| 2x2 stride, outputs 5x5x16   					|
| Flatten				| outputs 400									|
| Fully Connected		| outputs 210									|
| RELU					| 	         									|
| Fully Connected		| outputs 128									|
| RELU					| 	         									|
| Fully Connected		| outputs 43									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training is in cell 13.

I used AdamOptimizer as the optimizer. Once i found that the model is learning, I gradually increased the epoch and see that the model appears to be still learning, so i ran for 100 epochs, with a batch size of 128. Similar to Lenet, I used a learning rate of 0.001. (see cell 10 and 11)




####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.935 
* test set accuracy of 0.921

Lenet architecture was used. Similar to Lenet, the images have been pre-processed to a specific dimensions. Once configured and was able to test that model is running and learning. It's accuracy is surprisingly high right in the beginning. The accuracy for the validation and test set is close.

Encountered hurdles on some points in this implementation. Primarily on the following:
1. pre-processing - turning the images into grayscale changes the shape of the numpy array and has to be reshaped
2. Classes for LeNet is not equal to number of Signs - played around with the Fully Connected layer outputs


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

German traffic signs are saved in https://github.com/angelovila/CarND-P2-traffic-sign-classifier/tree/master/web-signs

Right of the bat, the images have to be resized to be in the proper dimensions.

The first image has a another sign at the top part which may confuse the model. Other images appear to be at an angle. The third image has watermark which might affect the prediction, but when processed, the watermark isn't as obvious in the human eye.

These images are loaded in cell 15 to cells 20. Then processed in cells 21 to 24


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Out of the 5 images pulled from the web, it predicted 3 out of 5. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution 		| General Caution   							| 
| Pedestrians  			| Traffic Signals 								|
| Speed Limit 20km/h	| General Caution								|
| Speed Limit 20km/h	| Speed Limit 20km/h			 				|
| Keep Right			| Keep Right         							|


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is confident that the sign is General Caution and it's correct. Probabilities of other signs are below

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >0.99        			| General Caution   							| 
| >.01     				| Traffic Signals 								|
| >.01					| Children Crossing								|
| >.01	      			| Pedestrians				 					|
| >.01				    | Road narrows on the right      				|


For the second image, the model guessed Traffic Signals while it should've been Pedestrians. Probabilities of other signs are below.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .7        			| Traffic signals   							| 
| .1     				| Go straight or right 							|
| .07					| Priority Road									|
| >.01	      			| Right-of-way at the next intersection			|
| >.01				    | No Passing				      				|



For the third image, the model is confident that the sign is General Caution but the sign is actually 20km/h. Potentially, the watermarks on the original image somehow affected the model. Probabilities of other signs are below
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >0.99        			| General Caution   							| 
| >.01     				| Slippery Road		 							|
| >.01					| Go straight or left							|
| >.01	      			| Right-of-way at the next intersection			|
| >.01				    | Road narrows on the right     				|



For the fourth image, it correctly identified the speed limit of 20km/h sign. Probabilities of other signs are below
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >0.99        			| Speed limit (20km/h) 							| 
| >.01     				| Speed limit (70km/h)		 					|
| >.01					| Speed limit (30km/h)							|
| >.01	      			| Go straight or Right 							|
| >.01				    | Turn right ahead    							|


For the fifth image, the model is confident that the sign is Keep right.
Probabilities of other signs are below
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >0.99        			| Keep Right 		 							| 
| >.01     				| Yield		 									|
| >.01					| Dangerous curve to the right					|
| >.01	      			| Speed limit (50km/h)							|
| >.01				    | Speed limit (80km/h) 							|

