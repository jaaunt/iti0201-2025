import numpy as np

class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.found_object = False
        self.object_distance = 0.0

    def detect_blue_object(self) -> bool:
        """Detect a blue object using the robot's camera."""
        image = self.robot.get_camera_rgb_image()
        detected = self.process_for_blue(image)
        return detected

    def process_for_blue(self, image) -> bool:
        """Process the camera image to check for a blue object."""
        blue_detected = np.any(image[:, :, 0] > 100)
        return blue_detected

    def drive_towards_object(self) -> None:
        """Drive towards the detected blue object."""
        self.robot.set_left_motor_velocity(0.5)
        self.robot.set_right_motor_velocity(0.5)

    def stop_in_front_of_object(self) -> None:
        """Stop the robot in front of the blue object."""
        self.robot.set_left_motor_velocity(0)
        self.robot.set_right_motor_velocity(0)

    def search_for_object(self) -> None:
        """Turn the robot to search for the blue object."""
        self.robot.set_left_motor_velocity(-0.2)
        self.robot.set_right_motor_velocity(0.2)

    def sense(self) -> None:
        """Gather sensor data."""
        self.found_object = self.detect_blue_object()

    def plan(self) -> None:
        """Plan the robot's actions."""
        if self.found_object:
            self.object_distance = self.robot.get_lidar_range_list()
            if self.object_distance > 0.5:
                self.drive_towards_object()
            else:
                self.stop_in_front_of_object()
        else:
            self.search_for_object()

    def act(self) -> None:
        """Execute the actions."""
        pass

    def spin(self) -> None:
        """Main loop."""
        while True:
            self.sense()
            self.plan()
            self.act()
