
from model import Driver, ModelType
from data import TrainingData

"""The training of the CNN is performed in two steps. The first step runs with a limited training
set for 5 epochs with a learning rate of 0.001. This has been found to establish a solid foundation
that allows the car to make it all the way around the track, albeit in a rather slopy way. The second
step trains with a lower learning rate and much more data for an additional 5 epochs. This fills in 
the gaps in the CNN and smooths out the driving considerably.
"""

# build the driver
driver = Driver((80, 80, 1))
driver.build(ModelType.CONV1)

##################### Initial training ####################
initial_data = [
                    './data/trk1_normal_1', 
                    './data/trk1_normal_2', 
                    './data/trk1_normal_3', 
                    './data/trk1_corner_infill',
                    './data/udacity_data',
                ]

# 1) The initial training step
data = TrainingData(initial_data)
driver.trainGen(data.training_generator, 
                data.training_size, 
                5, 
                data.validation_generator, 
                data.validation_size,
                lr=0.001)

####################### Fine tuning #######################
fine_tune_data = [
                    './data/trk1_normal_1', 
                    './data/trk1_normal_2', 
                    './data/trk1_normal_3', 
                    './data/trk1_normal_4', 
                    './data/trk1_swerve_fill', 
                    './data/trk1_corner_infill',
                    './data/trk1_corner_infill_2',
                    './data/trk1_bridge_infill',
                    './data/trk1_corners',
                    './data/trk2_normal_1',
                    './data2/trk1_corrections_1',
                    './data2/trk1_corrections_2',
                    './data2/trk1_small_swerve',
                    './data2/trk1_small_swerve_2',
                    './data2/trk1_small_swerve_3',
                    './data2/trk1_normal_1',
                    './data/udacity_data',
                ]

# 2) The fine-tuning step
data = TrainingData(fine_tune_data)
driver.trainGen(data.training_generator, 
                data.training_size, 
                5, 
                data.validation_generator, 
                data.validation_size,
                lr=0.0001)

# write the model to disk
driver.save('model')