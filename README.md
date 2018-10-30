# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/visualization.png "Visualization"
[image2]: ./report_images/training_distribuition.png "Training dataset distribuition"
[image3]: ./report_images/original_vs_normal_gray.png "Grayscaling"
[image4]: ./report_images/web_images.png "Web images"
[image5]: ./report_images/top5softmax.png "Softmax"
[image6]: ./report_images/prediction_histogram.png "Predictions histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one..

You're reading it! and here is a link to my [project code](https://github.com/kadway/SelfDrivingCarND-Proj3/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate the summary statistics of the traffic signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the data set.

Here is an exploratory visualization of the data set.
It shows the first occurrence of each traffic sign type in the training data set.

![alt text][image1]

In the following histogram is shown how many of the classes/labels of each traffic sign type are present in the training data set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the input size to the neural network making it also easier to train.
For traffic sign classification the grayscaling seems to be a good approach of simplifing the data because the necessary features for classification should still exist in the grayscaled image.
After grayscaling I normalized the image with `normalize_grayscale(image_data)` taken from Udacity lessons. The function performs a Min-Max scaling to a range of [0.1 to 0.9].
The purpose is to have the value of the pixels closer together and have its mean to be close to zero.
This will facilitate and speed up the learning process of the neural network.
Here is an example of a traffic sign image before and after grayscaling and normalizing.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale normalized image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride 2x2 kernel, outputs 14x14x100 		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x200  |    									
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride 2x2 kernel, outputs 5x5x200 		|
| Convolution 1x1	    | 1x1 stride, valid padding, outputs 5x5x300    | 
| RELU                  |                                               |   
| Max pooling	      	| 2x2 stride 2x2 kernel, outputs 2x2x300        |
| Flattening            | output 1200		                            |
| Fully connected 1		| output 200    								|
| RELU                  |                                               |
| Dropout               | keep probability 50%                          |
| Fully connected 2		| output 43       					            |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the following set of hyperparameters gave a reasonable good validation accuracy.
I used a batch size of 100, Learning Rate 0.0009, L2_regularization weight of 1e-06, keep probability of 0.5 and 20 EPOCHs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.969
* test set accuracy of 0.961

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

**The first architecture used was LeNet provided from Udacity lessons. It helped to get an understanding
of how to train the model by adjusting the hyperparameters.**

* What were some problems with the initial architecture?

**The initial architecture was not producing a validation accuracy higher than 93%, even after much tweeking of the hyperparameters.**

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In order to improve accuracy I started to extend the model with one more convolution layer after the existing 2 and then experiment with the sizes of the filters for each layer.
The result was clearly visible and the accuracy was increasing much. But the learning process was never improving much more and stopping around 94% and the model was now much more slow to learn.
I decided to try and remove the last fully connected layer which reduced the the model size speeding up the learning process. 
The accuracy did not change that much and it was still possible to get around 94% accuracy after tweeking one more time the size of the layers.
To try and increase the accuracy past 94% I added a dropout layer after the first fully connected layer and also added an L2 regularization to my loss function. 

* Which parameters were tuned? How were they adjusted and why?

The learning rate and batch size were the first parameters to be adjusted. I went for a slower learning rate and it provided better end results as compared to a higher learning rate.
The batch size was also adjusted and it seems that when too high or too low it would reduce the accuracy considerably. I started with a batch of 128 and did variations of batches above and bellow this number, end up with a batch of 100.
The dropout layer was tested with different keep probabilities and at 50% was found to be improving the validation accuracy.
By adding this layer I also prevent over-fitting the model because it can never be sure to "memorize" very specific features from the training set, making it more flexible to predict on real world features.
The L2 factor was also played with and mostly it was making the accuracy worst when the factor was between 0.1 and 1e-05.
At 1e-06 I noticed that the accuracy was back to the values before applying L2.
But additionally the model was more stable while learning, the accuracy would increase with each epoch but not too much at a time.
Before L2 it would happen that the accuracy would increase considerably in one epoch but in the next epoch it would also decrease a fair amount showing that after a certain accuracy value 93%-94%, the learning was not linearly increasing as desired.
So L2 seems to have contributed, even if only a little, to penalise the heavier weights in the model and make the learning smoother and more linear when close to the achieved accuracy limit. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Using convolution as the first layers is important because it looks at the input and does not destroy the spacial data. 
It is also less expensive to compute and requires less memory as compared to the fully connected layers.
In each convolutional layer the weights and bias are shared and each patch of pixels connects to a neuron in the next layer.
Each patch will connect to n neuron in the next layer defined by the filter depth.

* Why did you believe it would be relevant to the traffic sign application?

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![alt text][image4] 

bla bla 
![alt text][image5] 

bla bla
![alt text][image6]



The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



generation of Additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:
