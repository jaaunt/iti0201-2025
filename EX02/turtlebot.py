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
    """
    Turtlebot robot class.
    """

    def __init__(self):
        """
        Initialize robot object.

        Args:
            time = the time when the activity occurred.
            orientation = the orientation of the robot.
            range_list = a list of measured distances.
            enc_l = power of the left engine.
            enc_r = power of the right engine.
        """
        self.time = None
        self.orientation = None
        self.range_list = None
        self.enc_l = None
        self.enc_r = None

    def get_time(self):
        """
        Return the current time.

        Returns:
            the current time.
        """
        return self.time

    def get_orientation(self):
        """
        Return the current orientation.

        Returns:
            the current orientation.
        """
        return self.orientation

    def get_lidar_range_list(self):
        """
        Return the current lidar range list.

        Returns:
            the current lidar range list.
        """
        return self.range_list

    def get_left_motor_encoder_ticks(self):
        """
        Return the current left motor.

        Returns:
            the current left motor power.
        """
        return self.enc_l

    def get_right_motor_encoder_ticks(self):
        """
        Return the current right motor.

        Returns:
            the current right motor power.
        """
        return self.enc_r

    def _set_data(self, data_at_time_step: list) -> None:
        """
        Format the data.
        """
        self.time = data_at_time_step[0]
        self.orientation = data_at_time_step[1]
        self.enc_l = data_at_time_step[2]
        self.enc_r = data_at_time_step[3]
        self.range_list = data_at_time_step[4]
