# Behavioral-Cloning

### Description:
In this project I demonstrate the creation of a convolutional neural network that is capable of steering a vehicle around a track using only images from a center mounted dash cam. This project was part of the [Udacity Self-Driving Car Engineering Nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

#### Code organization:
- [model.py](./model.py): Defines and build the convolutional nueral networks.
- [process.py](./process.py) : Performs preprocessing on the dash-cam images.
- [data.py](./data.py): Manages the data for training, testing and verification.
- [model.h5](./model.h5): The trained and saved keras model.
- [drive.py](./drive.py): Responsible for communicating with the simulator and running dash-cam images through the network to produce the next steering angle.

### References:
The following sources provided ideas and inspiration for this project:

- [Udacity self-driving-car](https://github.com/udacity/self-driving-car)
- [NVIDIA End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)
- [Keras VGG16](https://keras.io/applications/)
- [Michael Nielsen's Deep-Learning Book](http://neuralnetworksanddeeplearning.com/chap3.html)
- [VGG in Tensorflow](https://www.cs.toronto.edu/~frossard/post/vgg16/)
- [ImageNet](http://www.image-net.org)

### Pipeline:

#### 1) Gathering training data:
Training data for this project was collected using a simulator provided by Udacity. The simulator is essentially a video game that allows the player to drive a car around a track and optionally record data along the way. The simulator includes 2 tracks that each feature a single lane road that winds it's way through various road surfaces, lane markers, bridges, landscapes and realistic distractions such as leaves and debris on the roadway. During recording, sample points are collected at a rate of 10 Hz. Each sample point consists of an image from each of the 3 virtual cameras attached to the car as well as information describing the cars instantaneous speed and steering angle. The virtual cameras are designed to mimic fender mounted cameras on the drivers and passengers side as well as a center mounted dash-cam. When a recording session is completed, a log file is produced in comma-separated format that includes columns for each instantaneous value as well as each of the three image file paths.

A total of 18 different recording sessions were created and used to train the network. During the first few recordings, the car was driven down the center of the track as much as possible, similar to how a well behaved human would ideally drive the vehicle. However, it was observed that when the network which was trained with only these 'happy path' examples was put to the test and asked to steer the car autonomously, the car would rather quickly drift off center and eventually veer off the road. The reason for this is that the network has not seen any examples of what to do when the car has drifted off course and therefore is not capable of correcting the situation. To remedy this, several recording sessions were made with the intent of providing examples of course correction. The following strategies were used in various degrees:

- Gently swerving from one side of the road to the other while staying within the bounds of the lane. 
- Intentionally positioning the car in an undesirable situation with recording mode turned off and then toggling it on as the car corrects the situation by steering back to center. 
- Driving the entire track backwards to help balance out the number of left turns and right turns.  
 
When all of these training sets were put together, the result was a network that was able to safely navigate the car through the entire track.

#### 2) Preprocessing:
In general, preprocessing the input images before feeding them into a network is a key step in achieving successful results. However, in this project I found that I was able to achieve decent results with very little preprocessing at all. I believe that this is mainly due to the fact that the simulator provides a very controlled and repeatable environment in which the images generated are of high quality with very little variance in exposure, focus, brightness etc. 
 
The shape of the images that are fed to the network depends on the chosen architecture. The VGG16 based network is restricted to accepting images with 3 channels whereas the more basic architecture is able to accept grayscale images. I found that either format was able to produce acceptable results. However, I believe that in this particular case, given the simulator with only two tracks to collect data from, training with color images may lend itself to overfitting more so than with grayscale images.  
 
The thing that I found to be the most critical to the performance of the network was the size of the images. The simulator produces images that are 320x160 pixels in size and I found that working with the images at this size was challenging due to the large memory demands that are created from such a large input layer. I also found that running the pipeline with images of this size did not improve the outcome and in fact the opposite effect was observed in several cases. I believe the reason for this is that the images of road seen from the dash cam are relatively sparse in useful information. The landscape surrounding the track, the hood of the car, etc. all make up a large part of the images but do not contribute to the underlying behaviors that we want the network to learn. In other words, these relatively large resolutions increase the size of the network without substantially increasing the amount of information available to the it.  
 
To help increase the signal to noise ratio of the training images I decided to crop off the top 40%. This effectively removes the section of image above the horizon while retaining nearly all of the useful information from the track, walls and lane lines. This step alone led to a great improvement in driving style. After cropping, the images were resized to 80x80 pixels and normalized to a range of [-0.5, 0.5].

<br />
![Results of the preprocess step](/Images/preprocess.png)

*Fig. 1 - left: An unaltered image from a recording session. right: The same image after going throught the preprocessing step*
<br />
<br />

#### 3) Network architecture and training:

In this project I built and tested two convolutional networks. The first is simple and relatively small, consisting of 3 convolutional blocks followed by 2 fully connected layers and a total of 209,301 trainable parameters.

The second is based on a pre-trained version of the popular VGG16 architecture (see image below) with a custom top section added on for fine tuning. This network is much larger indeed, weighing in at a total of 18,914,689 parameters, 4,200,001 of which are trainable.

<br />

##### - VGG16:

[![VGG16 CNN architecture](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)](https://www.cs.toronto.edu/~frossard/post/vgg16/)

*Fig. 2 - A visual diagram of the VGG16 network architecture. [Source: VGG in Tensorflow](https://www.cs.toronto.edu/~frossard/post/vgg16/)*

<br />

The VGG16 model has been previously trained on the [ImageNet](http://www.image-net.org) dataset which contains images sized to 224x224x3. However, as discussed above, the input images for this project are sized at 80x80x3 and because of this it was required that we exclude the fully connected layers when importing the pre-trained model. After this step we are left with the 5 fully trained convolutional blocks that act as a base to build a custom regression model. I experimented with several configurations for the new top end but in the end I found good results from a relatively small number of wide fully connected layers with max-pooling and ReLU activations (See Fig. 3 below). 

<br />
![Custom VGG16 top-end for regression task.](/Images/vgg16Regression.png)

*Fig. 3 - Custom VGG16 top-end for regression task*
<br />

I trained this model with an Adam optimizer with learning rates ranging from 1e-5 to 1e-2 and found that the sheer size of the model made it very easy to overfit the relatively small training set, even when training for a single epoch. To alleviate this issue, I introduced 2 dropout layers with keep-probabilities of 0.5 and found that this architecture combined with a learning rate of 1e-3 produced decent results.  

<br />
##### - BASIC:

After playing with the VGG16 model discussed above, I started to wonder if a smaller model architecture would be able to produce similar results. A smaller model would be beneficial in several ways including a somewhat lesser tendency to overfit the limited training data, increased training speed and most importantly increased prediction speed. If the network was to be used in a real vehicle (as opposed to a simulator) then the speed at which the network is able to predict the next steering angle would be critically important. A pipeline that takes too long could be a show stopper. 
 
To this end I created and trained the network architecture shown in fig. 4. Once again I used the Adam optimizer and trained in multiple stages with a decreasing learning rate at each stage. To my surprise the results produced by this smaller network were just a good if not better than with the VGG16 based model. 
 

<br />
![Basic model architecture.](/Images/basicArcitexture.png)

*Fig. 4 - Basic model architecture for regression task*
<br />

After all of training was complete. The models were verified against a separate dataset that was collected for this purpose. This dataset was not included in the training steps so that it would serve as a good measure of the networks ability to generalize. The verification step was able to ahieve a loss of well under 0.1 which is very good.


### Result:

The result of the pipeline discussed above is a convolutional neural network that is capable of navigating a car throughout the entire track smoothly and safely (please see video below).

[![Success video link](/Images/successVideoThumbnail.png?raw=true)](https://youtu.be/_oy2a1zdvNk "Behavioral cloning success!")

### What's next

The network that came out of this project is functional but far from perfect. There are several places in the course that the car drifts a little closer to the edge of the lane than I would like and the driving style could be smoother. I think that in order to get there I will need to generate more training data or get more creative with the data that I already have. Some possibilities are: 
 
- Generate more data by driving in the simulator 
- Horizontally flip the training images that I have and negate the steering angles. This would effectively double my training data. 
- Generate new training samples by randomly perturbing the ones that I have. The associated steering angles would also need to be perturbed in a way that makes sense. 
 
Another thing that I would like to experiment with would be to use the network to predict the vehicle speed as well as the steering angle. All of the required ingredients for this work are already here so it's just a matter of doing it. 
 
