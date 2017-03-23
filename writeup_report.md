#**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode. This file was modified to suit my network.
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py, model.h5 and the drive.py files submitted is sufficient to run the simulator. The drive.py file was modified to crop and resize frames as expected by the model. The drive.py was further modified to change the throttle value during sharp streering angles. This modification helps smooth the ride on slow computers.

###Model Architecture and Training Strategy

####1.  Convolution Neural Network Model Architecture
The model consists of a convolution neural network made of the following layers (line 157 to 176):
1. One lambda layers to normalize the input
2. Multiple 2D convolution layers with 3x3 filters with depths ranging from 32 to 128 and strides of 2x2 and 1x1
3. Rectilinear activation (Relu) layers between the 2D convolution layers to introduce non linearity
4. Max pooling layers with kernel size of 2x2
4. Fully connected layers consisting 128, 64 and 1 output neuron.
5. Dropout layers with a keep probability of 0.5 between the fully connected layers to prevent over fitting
6. The model uses adam SGD optimizer with default parameters.

The model take an input tensor of shape (none,64,64,3). The training data and the frames output by the simulator is  of  the shape 320x160 with 3 channels for RGB . Hence, the data input to the model is cropped and resized to 64x64x3. 

#####2. Fit Generator
The training model uses a fit generator to feed training data set to the neural net model. This allows us to delay loading the images into memory until is it needed (line 124). The data_generator function continuously feeds batches of training data to the model for each epoch. 

####2. Attempts to reduce over-fitting in the model

The model attempts to prevent over fitting using dropout layer with a keep probability of 0.5

####3. Model parameter tuning

The model has several parameters that impact the performance. These are detailed below:

#####Optimizer parameter:
The model uses adam optimizer with the following default parameters:
lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0

#####Generator parameters
I used the following parameter settings for the fit generator :
samples per epoch=7936
batch size=256
number of epochs =15
#####Steering angle shift per pixel:
The training data was augmented by randomly shifting the image left or right and adjusting the corresponding steering angle by +- 0.0025.  

#####Steering angle shift between Left and Right camera image
The training data set was augmented by using randomly selecting the left and right camera images. The steering angle was adjusted by adding or subtracting 0.25
 

####4. Appropriate training and validation data
I used the raw data provided by Udacity. The training and validation data set was created as follows:

##### Training strategy
After several attempts of trying to play with the simulator using my keyboard, I decided to use the training data set provided by Udacity. This data set has several recovery training scenarios. However, the data set has disproportionately large samples with zero steering angle.  By trail and error, I found that this has negative effect on the performance in the simulator. So, I formulated a training strategy that will use all of the non-zero samples and only 15% of the zero training samples. In addition, I would augment only the zero angled training sample by horizontal shifting and flipping to make it non-zero. Further, to add to the training data set, I included a random sample of the left and the right camera images with steering angle adjustment.

Here is a animation of the recovery training performed on the simulator

![Recovery training animation](https://github.com/neelks72/wavy-driver/blob/master/output_8Qfe3v.gif?raw=true)

##### Training data set formulation 

1. The training data was created from random sample of the raw data. I used 80% of the raw data for training. 

2. The raw data provided has a very large proportion of samples where the steering angle is zero. This will bias the training, hence I included only a random sample of about 700 with zero steering angle . The plot below shows the distribution of the steering angles

3.  A random selection of images from the left and right cameras was added to the training data set. The steering angle was compensated by adding or subtracting 0.25 units (line 33 to 38). 

4. The training samples with zero steering angle was further augmented by shifting it to the right or left by randomly by up to 30 pixels. The steering angles was correspondingly adjusted by +-0.0025 per pixel shifted. This further reduced the number of training samples with zero steering angle. (line 64)

5. The sample was further randomly flipped and brightness altered (line 53 and  56)

Here is a histogram showing the steering angle distribution in the raw training data set

![Steering angle distribution before augmentation](https://github.com/neelks72/wavy-driver/blob/master/Before_Zero_Bias_Correction.png?raw=true)

Here is a histogram showing the steering angle distribution after data augmentation of training data set

![Steering angle distribution after augumentation](https://raw.githubusercontent.com/neelks72/wavy-driver/master/After_Zero_Bias_Correction.png)

 Here are some sample frames from the training data set and their corresponding steering angles before and after augmentation

![Training data set frames](https://github.com/neelks72/wavy-driver/blob/master/Augmentation.PNG?raw=true)

Here are some sample frames from the training data after it is cropped and resized to 64x64 pixels. 

![enter image description here](https://github.com/neelks72/wavy-driver/blob/master/Frames.PNG?raw=true&sds)
#####Validation data set formulation 
The validation data set was created by splitting the raw data such that 80% was used for training and 20% was set aside for cross validation (line 207), This resulted in 1823 samples set aside for validation.

####5. Pre-processing data

The pre-processing involved the following steps:

1. Crop the image to remove unwanted information such the sky and dashboard. I removed 70 pixels from the top and 20 pixels from the bottom of the image.
2. The image was scaled to 64x64 pixels.
3. The image was normalized to fall within a range of -0.5 to 0.5 so that the data set has zero mean and similar variance . This helps the optimizer in minimizing the loss. The normalization is done in the Keras lambda layer

####6. Modifying Drive.py
The drive.py script spins up a local server that receives the center camera image from the simulator and feeds the steering angle and throttle information to the simulator.  The steering angle is predicted using the model that was trained. Since the model expects the data 64x64 pixel format,  I crop and resize the image before it used to make the prediction. Further, to smooth en the ride, I modified the code to slow down the throttle when the model predicts high steering angles. 

###**Reflection**

The behavioral cloning was a great exercise.  The project has given some insights in the following areas:

Quality of training data:
The quality of the training data is probably the most important aspect for the performance of a model. I started off project using the training data that I generated by driving the car in the simulator in the middle avoiding the edges (minimal recovery). This created large data sets with zero steering angles. The model trained on this type of data was not useful as the car kept running off the road. The model should be trained on recovery 

Network architecture:
There is no one architecture that will work well for all scenarios and at the same time there can be more than one configuration of parameters and network layers that works reasonably well for a given problem.

Parameter sensitivity:
The performance of the model seems pretty sensitive to some of the assumptions I made such the steering angle offset for left and right camera as well the shift per pixel. I guess, if we know the exact geometry of the camera positions, we should be able to arrive at precise number.







