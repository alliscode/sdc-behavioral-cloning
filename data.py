
from process import read_image, pre_process
from keras.models import load_model
import numpy as np
import os.path
import cv2

class TrainingData:
    """A class to represent and manage the training data for the CNN.
        
    Args: 
        training_sets: An list of paths to training sets. Each path must point to a directory
        that contains a driving_log.csv file as well as a directory of trianing images.
        batch_size (optional): The batch size that will be yielded by the generators. Defaults to 128
        validation_split (optional): The portion of data to split out for validation.
    
    Attributes:
        train_generator: A generator that yields a collection of training images and a collection
        training labels.
        validation_generator: A generator that yields a collection of valiation images and a collection
        validation labels.
        training_size: The number of training samples.
        validation_size: The number of validation samples
        training_sets: An list of paths to training sets. Each path must point to a directory
        that contains a driving_log.csv file as well as a directory of trianing images.
        batch_size (optional): The batch size that will be yielded by the generators. Defaults to 128
        validation_split (optional): The portion of data to split out for validation.
    """
    
    def __init__(self, training_sets, batch_size=128, validation_split=0.2):
        self.training_sets = training_sets
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.training_generator = None
        self.validation_generator = None
        self.training_size = None
        self.validation_size = None
        self.load_data()

    def load_data(self):
        """Loads the provide training sets by creating generators for the test and validation sets."""
        
        self.sample_maps = []

        # merge all of the sample maps
        for directory in self.training_sets:
            mapFile = directory + '/driving_log.csv'
            images = directory + '/IMG'

            # validate the input
            if not os.path.isdir(images) or not os.path.isfile(mapFile):
                raise ValueError("parameter must be a valid directory containing images and csv map.")

            # read the contents of the map file
            with open(mapFile) as f:
                self.sample_maps.extend(f.readlines())

        all_samples = np.arange(0, len(self.sample_maps))
        np.random.shuffle(all_samples)
        validation_split_index = int(len(all_samples) * self.validation_split)
        validation_samples, train_samples = all_samples[:validation_split_index], all_samples[validation_split_index:]
        
        self.training_generator = self.generator(train_samples)
        self.validation_generator = self.generator(validation_samples)
        self.validation_size = len(validation_samples)
        self.training_size = len(train_samples)
    
    def generator(self, samples):
        """A generator used to feed data into the model fit routine.
        
        Args:
            sample: A list of samples defining the image and labels to use.
            
        Yeilds: A tuple containing a collection of images and a collection of associated labels.
        """
        
        num_samples = len(samples)
        while True:
            batch_X = []
            batch_y = []

            # shuffle up the images
            np.random.shuffle(samples)
            for i, sample_index in np.ndenumerate(samples):
                center, left, right, steering_angle, throttle, brake, speed = tuple(self.sample_maps[sample_index].split(','))
                steering_angle, throttle, brake, speed = float(steering_angle), float(throttle), float(brake), float(speed)
                if speed > 1.0:
                    center_image = pre_process(read_image(center))
                    batch_X.append(center_image)
                    batch_y.append(steering_angle)

                    if len(batch_X) % self.batch_size == 0 or i == (num_samples-1):
                        yield (np.array(batch_X), np.array(batch_y))
                        batch_X = []
                        batch_y = []