"""
To use this helper file, please add it to the same directory as your EX02.py file along with the data files.

The directory should have the following files in it:
.
├── EX02.py
├── turtlebot.py
├── {Data_file}.pkl
└── local.py

To use with data file set the value in local.py main to the task related .pkl file you want to use.
"""

import pickle
import EX02 as student
import turtlebot as turtlebot


def load_dataset(filename):
    """Load the dataset."""
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


if __name__ == "__main__":
    ex_data = load_dataset("long.pkl")
    turtlebot_interface = turtlebot.Robot()
    student_robot = student.Robot(turtlebot_interface)
    for i, data_at_time_step in enumerate(ex_data):
        turtlebot_interface._set_data(data_at_time_step)
        student_robot.spin()
        lidar_ranges = turtlebot_interface.get_lidar_range_list()
