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


class Robot:
    def __init__(self):
        """Initialize robot object."""
        self.time = None
        self.orientation = None
        self.range_list = None
        self.enc_l = None
        self.enc_r = None

    def get_time(self):
        """Return the current time."""
        return self.time

    def get_orientation(self):
        """Return the current orientation."""
        return self.orientation

    def get_lidar_range_list(self):
        """Return the current lidar range list."""
        return self.range_list

    def get_left_motor_encoder_ticks(self):
        """Return the current left motor."""
        return self.enc_l

    def get_right_motor_encoder_ticks(self):
        """Return the current right motor."""
        return self.enc_r

    def _set_data(self, data_at_time_step: list) -> None:
        """Format data."""
        self.time = data_at_time_step[0]
        self.orientation = data_at_time_step[1]
        self.enc_l = data_at_time_step[2]
        self.enc_r = data_at_time_step[3]
        self.range_list = data_at_time_step[4]
