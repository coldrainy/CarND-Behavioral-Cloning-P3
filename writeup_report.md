**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I just use the nvidia's model the class recommended.The model include 5 convolution layers with activation "relu" .And 4 fully connected layers.(line 108-117).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 6-27). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 119).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, more data from the place where the car got out of the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use the LeNet. I thought this model might be appropriate because I have used this model successfully distinguish the traffic sign.So it might also be useful in this project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. But the car didn't perform well.I gathered more data to train the model,but there was not much progress. 

So I decide to change the model to the nvidia's model.At first ,I got a strange graph about the trainning loss like bellow.

![image1](/graph/loss_graph_before.png)

After asking people in the slack,I realized that I made a mistake in the 'fit_generator' function.I didn't change the 'setps_per_epoch' after I used other two camera's data and used the flip function to augument the data.

After fixing it up,I got a good result similar to this bellow.And this mistake really confused me for a lot of days and made me delay for the recommand dead line.

![image2](/graph/loss_graph_after.png)

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Well in fact I have tried to add resize and grayscale layers to the model.But because I am not so familiar with the keras,and there isn't too much time so I did't add the layer successfully.So the 

architecture is the same as the nvidia's model.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image3](/graph/center.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![image4](/graph/side_to center1.jpg)
![image5](/graph/side_to center2.jpg)
![image6](/graph/side_to center3.jpg)


After the collection process, I had about 30000 number of data points.  I then preprocessed this data by cropping the data to reduce the redundant information.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 . I used an adam optimizer so that manually training the learning rate wasn't necessary.
