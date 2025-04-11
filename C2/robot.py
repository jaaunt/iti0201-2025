from __future__ import annotations
import math
import numpy as np

class Robot:
    def __init__(self, robot: object) -> None:
        self.current_color = None
        self.detected_objects = []
        self.robot = robot
        self.has_faced_object = False
        self.state = "search"
        self.left_velocity = 0
        self.right_velocity = 0
        self.color_order = ["blue", "red", "yellow"]
        self.current_color_index = 0
        self.color_object_angles = []
        self.previous_time = 0.0
        self.search_timer = 0.0
        self.max_search_duration = 10.0
        self.scanning_objects = []
        self.scan_start_angle = None
        self.scan_current_best = None
        self.rotation_complete = False
        self.target_angle = 0.0

    def sense(self) -> None:
        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()
        self.left_motor_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_motor_ticks = self.robot.get_right_motor_encoder_ticks()
        self.lidar_object_detection()
        if self.state in ["search", "scanning"]:
            self.image = self.robot.get_camera_rgb_image()
            self.fov = self.robot.get_camera_field_of_view()
            self.current_color = self.color_order[self.current_color_index]
            self.color_object_angles = self._get_color_object_angles(self.current_color)
            if self.state == "search":
                self.handle_no_colour()

    def lidar_object_detection(self):
        if self.lidar is None:
            self.range_list = []
            return
        else:
            self.range_list = self.lidar

        if not self.range_list or not isinstance(self.range_list, list):
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
                    objects.append((self.range_list[center_idx], self._get_angle(center_idx)))
                in_object = False

        self.detected_objects = self._filter_objects(objects)

    def _get_color_object_angles(self, color: str):
        if self.image is None or self.fov is None:
            return []
        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50
        if color == "blue":
            mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        elif color == "red":
            mask = (red_channel > green_channel + threshold) & (red_channel > blue_channel + threshold)
        elif color == "yellow":
            mask = (red_channel > blue_channel + threshold) & (green_channel > blue_channel + threshold)
        else:
            return []
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
        labled_mask = np.zeros_like(mask, dtype=np.uint32)
        lable_id = 1
        to_visit = []
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for y, x in np.argwhere(mask):
            if labled_mask[y, x] == 0:
                labled_mask[y, x] = lable_id
                to_visit.append((y, x))
                while to_visit:
                    current_y, current_x = to_visit.pop()
                    for dy, dx in neighbours:
                        new_y, new_x = current_y + dy, current_x + dx
                        if 0 <= new_y < height and 0 <= new_x < width:
                            if mask[new_y, new_x] and labled_mask[new_y, new_x] == 0:
                                labled_mask[new_y, new_x] = lable_id
                                to_visit.append((new_y, new_x))
                lable_id += 1
        return labled_mask, lable_id - 1

    def get_objects_range_list(self) -> list | None:
        return self.detected_objects if self.detected_objects else None

    def _get_angle(self, index):
        num_points = len(self.range_list)
        fov = 2 * math.pi
        angle_per_step = fov / num_points
        return index * angle_per_step

    def _filter_objects(self, objects):
        min_distance_threshold = 0.2
        valid_objects = []
        for obj in objects:
            if obj[0] > min_distance_threshold:
                valid_objects.append(obj)
        return valid_objects

    def plan(self) -> None:
        state_actions = {
            "search": self._handle_search,
            "scanning": self._handle_scanning,
            "approaching": self._handle_approaching,
            "fixing_trajectory": self._handle_fixing_trajectory,
            "finished": self._handle_finished,
        }
        if self.state in state_actions:
            state_actions[self.state]()

    def _handle_search(self):
        self.left_velocity = -2.0
        self.right_velocity = 2.0
        if self.color_object_angles:
            self.state = "scanning"
            self.start_scanning_mode()

    def start_scanning_mode(self):
        self.scanning_objects = []
        self.scan_start_angle = self.robot.get_gyro_angle()
        self.scan_current_best = None
        self.rotation_complete = False

    def _handle_scanning(self):
        self.left_velocity = -1.0
        self.right_velocity = 1.0
        current_angle = self.robot.get_gyro_angle()
        if abs(current_angle - self.scan_start_angle) >= 2 * math.pi:
            self.rotation_complete = True
            self.left_velocity = 0
            self.right_velocity = 0
            if self.scan_current_best:
                self.target_angle = self.scan_current_best[1]
                self.state = "fixing_trajectory"
            else:
                self._next_color()
                self.state = "search"
            return
        angles = self._get_color_object_angles(self.current_color)
        if not angles:
            return
        closest = None
        min_distance = float('inf')
        for angle in angles:
            if -0.1 < angle < 0.1:
                center_index = 480
                span = 9
                front_values = self.range_list[center_index - span: center_index + span + 1]
                valid = [d for d in front_values if d is not None and d != float('inf')]
                if valid:
                    distance = min(valid)
                    if distance < min_distance:
                        min_distance = distance
                        closest = (distance, self.robot.get_gyro_angle())
        if closest:
            self.scan_current_best = closest

    def _handle_approaching(self):
        self.left_velocity = 2.5
        self.right_velocity = 2.5
        if not self.range_list:
            return
        center_index = 480
        span = 9
        front_values = self.range_list[center_index - span: center_index + span + 1]
        valid = [d for d in front_values if d is not None and d != float('inf')]
        if valid and min(valid) < 0.4:
            self.state = "finished"

    def _handle_fixing_trajectory(self):
        current_angle = self.robot.get_gyro_angle()
        if current_angle < self.target_angle - 0.05:
            self.left_velocity = -0.4
            self.right_velocity = 0.4
        elif current_angle > self.target_angle + 0.05:
            self.left_velocity = 0.4
            self.right_velocity = -0.4
        else:
            self.state = "approaching"

    def _handle_finished(self):
        self.left_velocity = 0
        self.right_velocity = 0
        self.reset_detection_data()
        self.current_color_index = (self.current_color_index + 1) % len(self.color_order)
        self.search_timer = 0.0
        self.previous_time = self.robot.get_time()
        self.state = "search"

    def _next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.color_order)

    def handle_no_colour(self):
        current_time = self.robot.get_time()
        timestep = current_time - self.previous_time
        self.previous_time = current_time
        if not self.color_object_angles:
            self.search_timer += timestep
            if self.search_timer > self.max_search_duration:
                self._next_color()
                self.reset_detection_data()
                self.search_timer = 0
        else:
            self.search_timer = 0

    def reset_detection_data(self):
        self.detected_objects = []
        self.color_object_angles = []

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
