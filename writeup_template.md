# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/traffics.png "Visualization"
[image2]: ./images/stop_sign_proc.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/new_images.png "New Images"
[image5]: ./images/probabilty.png "Probabilty"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Horki/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is an representation of all 43 classes, one image per class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Preprocessing steps for images:

* Convert to grayscale
* Reshape into (32x32x1) image

Here is an example of a STOP traffic sign image before and after processing.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was same as in LeNet project example.

| Layer | Description |
|:-:|:-:|
| Input | 32x32x1 Grayscale image |
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 16x16x64 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 5x5x16 |
| Convolution 2x2 | 1x1 stride, valid padding, outputs 4x4x32 |
| RELU | |
| Fully connected | inputs 400, outputs 120 |
| Dropout RELU | |
| Fully Connected | inputs 120, outputs 84 |
| Dropout RELU | |
| Fully Connected | inputs 84, outputs 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer (same as in LaNet project).

* epochs = 100 (gave me 100% accurate results)
* learning rate = 0.001
* batch size = 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 93.5 %
* test set accuracy of 92.7 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first choose LaNet architecture because it is the only one that I am aware of so far.
* What were some problems with the initial architecture?
LaNet was my initial architecture.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I increased epochs to 100, and got 100% on prediction result.
* Which parameters were tuned? How were they adjusted and why?
Increased only epochs to 100.


### Test a Model on New Images

#### 1. Choose nine German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the [web](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news):

![alt text][image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image | Prediction |
|:-:|:-:|
| Speed limit (60km/h) | Speed limit (60km/h) |
| Speed limit (70km/h) | Speed limit (70km/h) |
| Priority road | Priority road |
| Yield | Yield |
| No entry | No entry |
| Road work | Road work |
| Turn left ahead | Turn left ahead |
| Ahead only | Ahead only |

The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.7 %.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all test images I got corresponding traffic sign by 100%.

![alt text][image5]
