import numpy as np

class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.found_object = False
        self.object_centered = False

    def detect_blue_object(self) -> bool:
        """Detect a blue object using the robot's camera."""
        image = self.robot.get_camera_rgb_image()
        detected, angle = self.process_for_blue(image)
        return detected, angle

    def process_for_blue(self, image) -> tuple:
        """Process the camera image to detect blue and calculate angle."""
        blue_pixels = np.where(image[:, :, 0] > 100)
        if blue_pixels[0].size > 0:
            angle = np.mean(blue_pixels[1]) - (image.shape[1] / 2)
            return True, angle
        return False, 0.0

    def spin(self) -> None:
        """Spin to find and center the blue object."""
        self.robot.set_left_motor_velocity(-0.2)
        self.robot.set_right_motor_velocity(0.2)

    def drive_forward(self) -> None:
        """Drive forward towards the object once centered."""
        self.robot.set_left_motor_velocity(0.5)
        self.robot.set_right_motor_velocity(0.5)

    def stop(self) -> None:
        """Stop the robot."""
        self.robot.set_left_motor_velocity(0)
        self.robot.set_right_motor_velocity(0)

    def sense(self) -> None:
        """Gather sensor data."""
        detected, angle = self.detect_blue_object()
        if detected:
            # Check if object is centered
            self.object_centered = abs(angle) < 10
        else:
            self.object_centered = False

    def plan(self) -> None:
        """Plan robot's actions."""
        if self.object_centered:
            self.drive_forward()
        else:
            self.spin()

    def act(self) -> None:
        """Execute planned actions."""
        pass

    def main_loop(self) -> None:
        """Main loop to sense, plan, and act."""
        while True:
            self.sense()
            self.plan()
            self.act()
