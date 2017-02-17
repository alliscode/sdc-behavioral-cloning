
"""This module can be used to build and train a convolutional nueral network used to predict
an appropriate steering angle for a self driving car based on images from a center mounted
dash cam.

Example:
    
    !) Build a model based on the VGG16 CNN
    2) Train it for 5 epochs
    3) Use it to make a prediction
    
    recorded_data = ... # array trianing data paths
    
    batch_size = 128
    trainGen, trainSize, valGen, valSize = getGenerators(recorded_data, batch_size)
    trainGenerator, validationGenerator = ...
    
    driver = Driver((224, 224, 3))
    driver.build(ModelType.VGG16)
    driver.trainGen(trainGen, trainSize, 5, valGen, valSize)
    
    driver.predict

"""

__author__  =  "Ben Thomas"

import os.path
import numpy as np
from enum import Enum
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.optimizers import Adam

class ModelType(Enum):
    """An enum type for specifying a model architexture."""
    
    CONV1 = 1,
    VGG16 = 2

class Driver:
    """"""
    
    def __init__(self, input_shape):
        """Constructs an instance of the Driver class.
        
        Args:
            input_shape (int, int, int): The shape of the images that will be used with the model.
            
        Returns:
            Driver: The initialized instance.
        """
        
        self.input_shape = input_shape
        
    def build(self, model_type):
        """Builds a convolutional neural network.
        
        Args:
            model_type (ModelType): The type of model to build.
        """
        
        if model_type == ModelType.CONV1:
            self.model = self.basicModel()
        elif model_type == ModelType.VGG16:
            self.model = self.vgg16()
            
    def load(self, model_path):
        """Loads the model at the provided path.
        
        Args:
            model_path: A path to a valid Keras model (.h5)
        """
        
        self.model = load_model(model_path)
        
    def __str__(self):
        """Provides a summary of the model."""
        
        if self.model is None:
            return "Model is not built."
        
        self.model.summary()
        return ""
        
    def basicModel(self):
        """A 'basic' convolutional neural network. This network contains 3 convolutional blocks
        and 2 fully connected layers followed by a dropout layer before the output neuron.
        
        Returns:
            keras.models.Model: The constructed model. The model returned from this 
            function has not yet been compiled.
        """
        
        # this model will be pretty basic with a single input, single output and no branching.
        # Because of this, the keras Sequential model will work fine
        model = Sequential()
        
        # convolution 1, max-pooling, relu for the nonlinearity
        model.add(Convolution2D(50, 15, 15, border_mode='valid', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        
        # convolution 2, max-pooling, relu for the nonlinearity
        model.add(Convolution2D(30, 10, 10, border_mode='valid'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        
        # convolution 3, max-pooling, relu for the nonlinearity
        model.add(Convolution2D(10, 7, 7, border_mode='valid'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        
        # flatten the filters before transitioning to the fully connected section
        model.add(Flatten())
        
        # fully connected 1 with 200 neurons, relu for the nonlinearity
        model.add(Dense(200))
        model.add(Activation('relu'))
        
        # fully connected 2 with 75 neurons, relu for the nonlinearity
        model.add(Dense(75))
        model.add(Activation('relu'))
        
        # dropout to help prevent over-fitting to the training data
        model.add(Dropout(0.25))
        
        # as currently stated, this is a regression problem. Because of this, the output layer is a single 
        # neuron with a continuous value and so softmax activations make no sense. Linear makes the most
        # sense in this case.
        model.add(Dense(1))
        model.add(Activation('linear'))
        
        return model
    
    def vgg16(self):
        """A more complex convolutional neural network. This network is based on the VGG16 CNN provided
        by Keras (https://keras.io/applications/). The pre-trained VGG16 model is followed up with a custom 
        top section to perform the regression task. The pre-trained VGG16 layers are frozen so training this
        model will only effect the top section.
        
        Returns:
            keras.models.Model: The constructed model. The model returned from this 
            function has not yet been compiled.
        """
        
        # The model architecture here is too complicated to use the Sequential model provided by Keras
        # so the functional API is used instead (https://keras.io/getting-started/functional-api-guide/).
        # This makes it easy to add a regression section on top of the pre-trained VGG16 model.
        input_layer = Input(shape=self.input_shape)
        
        # Create the VGG16 model. We don't include the top (fully connected layers) because that would
        # lock us in to using images with a shape of 224x224x3 as was done in the original VGG16 training.
        # We will provide our own fully connected section. (https://keras.io/applications/#vgg16)
        vgg16_trained = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
        
        # we want the freeze all of the layers of VGG16 to take adavantage of the pre-trained weights. 
        # Then we can train our custom top section to perform the regression task.
        for layer in vgg16_trained.layers:
            layer.trainable = False
        
        # build the model using Keras functional API. The output of the VGG16 convolutional sections are 
        # flattened before transitioning to the fully connected layers. Relu activations are used at each
        # step to introduce nonlinearity. Furthermore, dropout is added at several points to help prevent
        # this very large model from over-fitting our relatively small dataset.
        model = vgg16_trained.output
        model = Flatten()(model)
        model = Dense((1500))(model)
        model = Dropout(0.5)(model)
        model = Activation('relu')(model)
        model = Dropout(0.5)(model)
        model = Dense((750))(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dense(1)(model)
        model = Activation('linear')(model)
        return Model(input=input_layer, output=model)
    
    def train(self, X, y, batch_size=128, nb_epoch=5, lr=0.001):
        """Train the model with provided samples and labels. The training is performed using the Adam
        optimizer and the mean-squared-error loss function.
        
        Args:
            X: The training smamples
            y: The training labels
            batch_size (optional): the size of training batches. Defaults to 128
            nb_epoch (optional): the number of epochs to train for. Defaults to 5
            lr (optional): The learning rate used in the Adam optimizer. Defaults to 0.001
        """
        
        optimizer = Adam(lr=lr)
        self.model.compile(optimizer, loss='mse')
        history = self.model.fit(X, y, batch_size, nb_epoch)
    
    def trainGen(self, train_generator, sample_per_epoch, nb_epoch, validation_generator, validation_size, lr=0.001):
        """Train the model with provided data generators. The training is performed using the Adam
        optimizer and the mean-squared-error loss function.
        
        Args:
            train_generator: A generator that yields a batch of training images and labels
            sample_per_epoch: The number of samples/labels in each epoch
            nb_epoch: The number of epochs to train for. Defaults to 5
            validation_generator: A generator that yields a batch of validation images and labels
            validation_size: The number of samples used in the validation step.
            lr (optional): The learning rate used in the Adam optimizer. Defaults to 0.001
        """
        
        optimizer = Adam(lr=lr)
        self.model.compile(optimizer, loss='mse')
        history = self.model.fit_generator(train_generator, sample_per_epoch, nb_epoch, validation_data=validation_generator, nb_val_samples=validation_size)

    def predict(self, image):
        """Uses the already trained model to predict the appropriate steering angle for a given input image.
        Args:
            image: The image that the prediction is based on. The dimensions of the image must match the
            dimensions that the network was trained on.
            
        Returns (float) The predicted steering angle.
        """
        return float(self.model.predict(image, batch_size=1))
    
    #predict_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False)
    def predictGen(self, generator, num_samples):
        """Uses the already trained model to predict the appropriate steering angles for a batch of input 
        images produced by a generator.
        
        Args:
            generator: The generator that produced the input images
            dnum_samples: The number of examples to predict.
            
        Returns (float) The predicted steering angle.
        """
        
        return self.model.predict_generator(generator, num_samples)
    
    def save(self, file):
        """Saves the trained model to disk.
        
        Args:
            file (string): The name of the samed file. An extension of 'h5' will be appended to the file name if
            it is not already.
        """
        
        if os.path.splitext(file)[1] != 'h5':
            file = file + '.h5'
            
        self.model.save(file)