"""C1 updated for C3."""
from __future__ import annotations
import math
import numpy as np


class Robot:
    def __init__(self, robot: object) -> None:
        self.detected_objects = []
        self.robot = robot
        self.has_faced_object = False
        self.state = "search"
        self.left_velocity = 0
        self.right_velocity = 0

    def sense(self) -> None:
        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()
        self.left_motor_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_motor_ticks = self.robot.get_right_motor_encoder_ticks()
        self.lidar_object_detection()
        if self.state == "search" or self.state == "approaching":
            self.image = self.robot.get_camera_rgb_image()
            self.fov = self.robot.get_camera_field_of_view()
            self.blue_object_angles = self._get_blue_object_angles()

    def lidar_object_detection(self):
        if self.lidar is None:
            self.range_list = []
            return
        else:
            self.range_list = self.lidar

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
                    objects.append((self.range_list[center_idx], self._get_angle(center_idx)))
                in_object = False

        self.detected_objects = self._filter_objects(objects)

    def _get_blue_object_angles(self):
        if self.image is None or self.fov is None:
            return []

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50

        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        labeled_mask, label_count = self._find_blobs(mask)

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
        return angles

    def _find_blobs(self, mask):
        height, width = mask.shape
        labeled_mask = np.zeros_like(mask, dtype=np.uint32)

        label_id = 1
        to_visit = []
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for y, x in np.argwhere(mask):
            if labeled_mask[y, x] == 0:
                labeled_mask[y, x] = label_id
                to_visit.append((y, x))
                while to_visit:
                    current_y, current_x = to_visit.pop()
                    for dy, dx in neighbours:
                        new_y, new_x = current_y + dy, current_x + dx
                        if 0 <= new_y < height and 0 <= new_x < width:
                            if mask[new_y, new_x] and labeled_mask[new_y, new_x] == 0:
                                labeled_mask[new_y, new_x] = label_id
                                to_visit.append((new_y, new_x))
                label_id += 1

        return labeled_mask, label_id - 1

    def _estimate_cube_distance(self):
        if self.image is None or not self.blue_object_angles:
            return None

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50

        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        labeled_mask, label_count = self._find_blobs(mask)

        if label_count == 0:
            return None

        for i in range(1, label_count + 1):
            pixels = np.column_stack(np.where(labeled_mask == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)

            width = x_max - x_min
            height = y_max - y_min

            aspect_ratio = height / width if width != 0 else 0
            if 0.8 < aspect_ratio < 1.2:
                return height  # just return pixel height now!

        return None

    def _get_angle(self, index):
        num_points = len(self.range_list)
        fov = 2 * math.pi
        angle_per_step = fov / num_points
        return index * angle_per_step

    def _filter_objects(self, objects):
        min_distance_threshold = 0.2
        return [obj for obj in objects if obj[0] > min_distance_threshold]

    def plan(self) -> None:
        state_actions = {
            "search": self._handle_search,
            "approaching": self._handle_approaching,
            "fixing_trajectory": self._handle_fixing_trajectory,
            "finished": self._handle_finished,
        }

        if self.state in state_actions:
            state_actions[self.state]()

    def _handle_search(self):
        self.left_velocity = -2.0
        self.right_velocity = 2.0
        print("SEARCHING for cube...")
        if self.blue_object_angles:
            if -0.05 < self.blue_object_angles[0] < 0.05:
                self.left_velocity = 0.0
                self.right_velocity = 0.0
                self.state = "approaching"
                print("Cube FOUND and centered!")

    def _handle_approaching(self):
        print("APPROACHING the cube...")
        self.left_velocity = 1.5
        self.right_velocity = 1.5

        pixel_height = self._estimate_cube_distance()

        if pixel_height is not None:
            print(f"Cube pixel height: {pixel_height}")
            if pixel_height > 100:  # tune this threshold to stop close enough
                self.state = "finished"
                print("I, FINISHED — close enough.")
        else:
            self.state = "search"
            print("Lost the cube, returning to search...")

    def _handle_fixing_trajectory(self):
        print("FIXING trajectory...")
        if self.detected_objects[0][1] < 4.65:
            self.left_velocity = -0.4
            self.right_velocity = 0.4
        elif self.detected_objects[0][1] > 4.7:
            self.left_velocity = 0.4
            self.right_velocity = -0.4
        else:
            self.state = "approaching"

    def _handle_finished(self):
        self.left_velocity = 0
        self.right_velocity = 0
        print("I, END(myself)")

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
