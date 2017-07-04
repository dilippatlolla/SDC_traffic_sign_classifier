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
[image0]: ./other_images/train_init.png "train data"
[image1]: ./other_images/train_data.png "train data"
[image2]: ./other_images/aug_data.png "augmented data"
[image3]: ./other_images/lecun_arch.png "samernet/LeCun architecture"
[image4]: ./other_images/hist_plot.png "hist plot"

[image5]: ./web_traffic_signs/random_11.png "hist plot"
[image6]: ./web_traffic_signs/random_12.png "hist plot"
[image7]: ./web_traffic_signs/random_18.png "hist plot"
[image8]: ./web_traffic_signs/random_33.png "hist plot"
[image9]: ./web_traffic_signs/random_34.png "hist plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dilippatlolla/SDC_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
The traffic sign dataset consists of 39,209 32×32 px color images that we are supposed to use for training, along with validation and testing data. Each image is 32 x 32 size belonging to one of the classes and is represented as [0, 255] integer values in RGB color space.

I used the numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the train data set.
![alt text][image0]

Visually it can be noted that, the images range from different color and contrast intensities and brightness as well.

Here is the histogram plot of the classes in the train data

![alt text][image4]
Based on the hist plot, we can conclude that the dataset is very unbalanced as some classes are underrepresented than others.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the traffic signs have similar color combinations and it has been established that colors do not give any significant feature to be picked up by the deep learning network. This also allows to reduce the numbers of channels in the input of the network without decreasing the performance. Thus I have converted the RGB data to gray scale.

As a second step, I normalized the image data to range of [0 1]. This is done to get all the input data on the same scale. If scales for different features are different, it can have an negative impact on the ability to learn. Standardized feature values ensure that all features are equally weighted in their representation.

As a third step, I have enhanced the contrast of the images as some images of the same class are completely dark(may be due to lighting or shadow). Histogram equalization enhances an image with low contrast, using a method called histogram equalization, which “spreads out the most frequent intensity values” in an image. I have utilized adaptive histogram equalization from [scikit-image](http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html) to enhance the contrast of the image dataset.

Here is an example of a traffic sign image before and during the various stages.
row 1: rgb, row 2: grayscale, row 3: normalized and row 4: histogram equalized
Here is an exploratory visualization of the train data set.

![alt text][image2]

The amount of data we have is not sufficient for a model to generalize well. So we augment data.
To add more data to the data set, I created another new dataset deriving from the original training dataset, composed by 34799 examples. In this way, I obtain 34799x2 = 69598 samples in the training dataset

I have used Keras [Image Data Generator](https://keras.io/preprocessing/image/) to derive new image set from the initial training dataset.
Basically, each image in the training dataset is applied with a rotation, a translation, a zoom and a shear transformation.

Here is an example of an augmented image and its corresponding preprocessed stage images:

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After giving an initial try with Lenet architecture. I achieved an accuracy of 90%. Then, have adapted Sermanet/LeCunn traffic sign classification journal article.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray scale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 400 	|
| RELU					|												|
| Flatten Relu output				|outputs 400												|
| Flatten prev Conv output					|outputs 400												|
| concatenate					|outputs 800												|
| Fully connected		| input 800, output 43        									|

Here is an overview of the architecture from the Samernet/LeCun paper

![alt text][image3]

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer (already implemented in the LeNet lab). The final settings used were:

    batch size: 128
    epochs: 40
    learning rate: 0.0009
    mu: 0
    sigma: 0.1
    dropout keep probability: 0.8

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 98%
* test set accuracy of 96%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I have used Lenet architecture to begin with as it can be easily adapted as it takes the input as 32x32 image same as the input train data.

* What were some problems with the initial architecture?
The accuracy stagnated at 90%. the initial filters count was low.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I tried basically three different architectures.
The first is the Lenet. the second id the LeCun/Samernet architecture. The third is a modified LeCun/Samernet with more filters in the initial convolution layers.

* Which parameters were tuned? How were they adjusted and why?
The parameters which I tuned included the learning-rate, the batch size, the number of epochs, and the network topology itself.
That's quite a lot of tuning and I feel that I've got a long way to go before I understand the interactions between these hyper-parameters. One of the first things I tried, was adding the pre-processing step which added approximately 2% of accuracy improvement. Adding normalization to the preprocessing did wonders to the Loss behavior.

Also trued dropout. I have applied the dropout to the fully connected layer. I would like to give a try in applying the dropout to the convolutional layers to see how it impacts

Other improvements included trying different preprocessing techniques and also other augmentations to improve dataset as well as balance the train data can be used. Due to time constraints I have not included them in this. I consider this as work still in progress and will keep working on updating and improve the accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because it is a bit warped. But since we have included an augmented dataset in the train dataset, the netwoek classified it fine.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way      		| Right-of-way  									|
| Priority road     			| Priority road 										|
| General caution					| General caution											|
| Turn right ahead	      		| Turn right ahead					 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			|  Right-of-way   									|
| .999     				| Priority road  										|
| .999					| General caution											|
| .999	      			| Turn right ahead					 				|
| .999				    |  Turn left ahead      							|

For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
