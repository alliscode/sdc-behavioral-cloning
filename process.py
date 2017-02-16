
import cv2
import numpy as np

def read_image(image_path):
    """Reads an image from disk into a numpy array.
    
    Args:
        image_path (string): The path of the image to be read.
        
    Returns: (numpy.Array): A numpy array containing the image data.
    """
    
    image = cv2.imread(image_path)
    return image

def pre_process(image):
    """Preprocess an image for use in the CNN.
    
    Args:
        image (numpy.array): The image to process.
    
    Returns: (numpy.Array): The processed image.
    """
    
    # discard the top of the image
    height = int(0.6 * image.shape[0])
    image = image[-height:-1,:,:]
    
    #resize to 80x80
    image = cv2.resize(image, (80, 80))
        
    # convert to gray scale floats
    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(shape[0], shape[1], 1).astype(np.float32)
    
    # scale the image
    image = image/255.0 - 0.5

    return image