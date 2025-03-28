from __future__ import annotations
import math
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.detected_objects = []
        self.robot = robot
        self.has_faced_object = False
        self.state = "init"
        self.left_velocity = 0
        self.right_velocity = 0

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()
        self.left_motor_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_motor_ticks = self.robot.get_right_motor_encoder_ticks()

        if self.lidar is None:
            print("Lidar data is NULL!")
            self.range_list = []
            return
        else:
            self.range_list = self.lidar

        if not self.range_list or not isinstance(self.range_list, list):
            print("Invalid or empty Lidar data, skipping sensing.")
            self.range_list = []
            return

        objects = []
        in_object = False
        start_idx = None

        min_cluster_size = 1
        distance_jump_threshold = 0.3

        for i in range(1, len(self.range_list)):
            prev = self.range_list[i - 1]
            curr = self.range_list[i]

            if curr is None or prev is None or curr == float('inf') or prev == float('inf'):
                in_object = False
                continue

            if not in_object and abs(curr - prev) > distance_jump_threshold and curr < prev:
                in_object = True
                start_idx = i

            elif in_object and abs(curr - prev) > distance_jump_threshold and curr > prev:
                if i - start_idx >= min_cluster_size:
                    center_idx = round(start_idx + (i - start_idx) / 2)
                    objects.append((self.range_list[center_idx], self.get_angle(center_idx)))
                in_object = False

        self.detected_objects = self.filter_objects(objects)
        print(self.detected_objects)
        print(self.time)

        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()

        self.blue_object_angles = self.get_blue_object_angles()

    def get_blue_object_angles(self):
        if self.image is None or self.fov is None:
            return []

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50

        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        labeled_mask, label_count = self.find_blobs(mask)

        if label_count == 0:
            return []

        height, width = self.image.shape[:2]
        angles = []

        for i in range(1, label_count + 1):
            pixels = np.column_stack(np.where(labeled_mask == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)
            x_center = (x_min + x_max) / 2

            angle = ((x_center - width / 2) / (width / 2)) * (self.fov / 2)
            angles.append(angle)
        print("Blue object angles:", angles)

        return angles

    def get_angle(self, index):
        num_points = len(self.range_list)
        fov = 2 * math.pi
        angle_per_step = fov / num_points
        return index * angle_per_step

    def filter_objects(self, objects):
        min_distance_threshold = 0.2
        valid_objects = []

        for obj in objects:
            if obj[0] > min_distance_threshold:
                valid_objects.append(obj)

        return valid_objects

    def handle_turning(self):
        if not self.detected_objects:
            self.state = "search"
            self.has_faced_object = False
            return

        _, lidar_angle = self.detected_objects[0]
        print(f"Turning to object at angle: {lidar_angle}")

        angle_margin = 0.3
        dead_zone = 0.2

        if lidar_angle < 4.7 - angle_margin:
            self.left_velocity = -0.5
            self.right_velocity = 0.5
            print("Turning left")
        elif lidar_angle > 4.7 + angle_margin:
            self.left_velocity = 0.5
            self.right_velocity = -0.5
            print("Turning right")
        elif abs(lidar_angle - 4.7) < dead_zone:
            self.left_velocity = 0
            self.right_velocity = 0
            self.has_faced_object = True
            self.state = "confirming_color"
            print("I, FACING OBJECT — READY TO CONFIRM COLOR")

    def handle_approaching(self):
        self.left_velocity = 1
        self.right_velocity = 1
        if self.detected_objects:
            if 4.65 > self.detected_objects[0][1] > 4.8:
                self.state = "fixing_trajectory"
        if self.detected_objects and self.detected_objects[0][0] < 0.3:
            self.state = "finished"
            print("I, FINISHED")
        elif not self.detected_objects:
            self.state = "search"
            print("fked up situation")
