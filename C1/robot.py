import numpy as np

class Robot:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.found_object = False
        self.object_centered = False
        self.blue_percentage_threshold = 0.8

    def detect_blue_object(self) -> bool:
        image = self.robot.get_camera_rgb_image()
        detected, angle, blue_percentage = self.process_for_blue(image)
        return detected, angle, blue_percentage

    def process_for_blue(self, image) -> tuple:
        blue_pixels = np.where(image[:, :, 0] > 100)
        total_pixels = image.shape[0] * image.shape[1]
        blue_pixels_count = len(blue_pixels[0])
        blue_percentage = blue_pixels_count / total_pixels
        if blue_pixels_count > 0:
            angle = np.mean(blue_pixels[1]) - (image.shape[1] / 2)
            return True, angle, blue_percentage
        return False, 0.0, blue_percentage

    def turn(self) -> None:
        self.robot.set_left_motor_velocity(0.2)
        self.robot.set_right_motor_velocity(-0.2)

    def drive_forward(self) -> None:
        self.robot.set_left_motor_velocity(0.5)
        self.robot.set_right_motor_velocity(0.5)

    def stop(self) -> None:
        self.robot.set_left_motor_velocity(0)
        self.robot.set_right_motor_velocity(0)

    def sense(self) -> None:
        detected, angle, blue_percentage = self.detect_blue_object()
        if detected:
            self.object_centered = abs(angle) < 10
            if blue_percentage >= self.blue_percentage_threshold:
                self.stop()
        else:
            self.object_centered = False

    def plan(self) -> None:
        if self.object_centered:
            self.drive_forward()
        else:
            self.turn()

    def act(self) -> None:
        pass

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
