"""C2: Robot logic for approaching nearest color-coded pole."""
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
        self.max_search_duration = 20.0
        self.best_target_angle = None
        self.best_target_distance = float('inf')
        self.scan_start_angle = None
        self.found_target = False
        self.arrival_time = None
        self.state_start_time = None

    def sense(self) -> None:
        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()
        self.left_motor_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_motor_ticks = self.robot.get_right_motor_encoder_ticks()
        self.orientation = self.robot.get_orientation()
        if self.state == "search" or self.state == "adjusting":
            self.image = self.robot.get_camera_rgb_image()
            self.fov = self.robot.get_camera_field_of_view()
            self.current_color = self.color_order[self.current_color_index]
            self.color_object_angles = self._get_color_object_angles(self.current_color)
            self.handle_no_colour()

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

    def plan(self) -> None:
        actions = {
            "search": self._handle_search,
            "approaching": self._handle_approaching,
            "adjusting": self._handle_adjusting,
            "driving": self._handle_driving,
            "waiting": self._handle_waiting,
            "finished": self._handle_finished,
        }
        if self.state in actions:
            actions[self.state]()

    def _handle_search(self):
        if self.scan_start_angle is None:
            self.scan_start_angle = math.degrees(self.orientation) % 360
            print("Started 360° scan")

        self.left_velocity = -1.5
        self.right_velocity = 1.5
        current_angle = math.degrees(self.orientation) % 360
        if self.color_object_angles:
            if -0.1 < self.color_object_angles[0] < 0.1:
                front_distance = self._get_front_distance()
                if front_distance < self.best_target_distance:
                    self.best_target_angle = current_angle
                    self.best_target_distance = front_distance
                    print(f"New best target at angle {current_angle:.2f}°, distance {front_distance:.2f}m")

        angle_diff = (current_angle - self.scan_start_angle + 360) % 360
        if angle_diff > 350 and self.best_target_angle is not None:
            print(f"Scan complete. Best target: {self.best_target_angle:.2f}° at {self.best_target_distance:.2f}m")
            self.state = "approaching"
            self.left_velocity = 0
            self.right_velocity = 0

    def _get_front_distance(self):
        center_index = 480
        span = 25
        front_values = self.lidar[center_index - span:center_index + span + 1]
        valid = [d for d in front_values if d is not None and d != float('inf')]
        return min(valid) if valid else float('inf')

    def _handle_approaching(self):
        current_deg = math.degrees(self.orientation) % 360
        angle_diff = (self.best_target_angle - current_deg + 540) % 360 - 180

        if abs(angle_diff) > 5:
            print(f"Rotating to target: Δ{angle_diff:.2f}°")
            self.left_velocity = -1.0 if angle_diff > 0 else 1.0
            self.right_velocity = 1.0 if angle_diff > 0 else -1.0
        else:
            print("Yaw alignment complete. Starting visual fine-tuning...")
            self.left_velocity = 0
            self.right_velocity = 0
            self.state = "adjusting"

    def _handle_adjusting(self):
        min_dist = 100
        cam_angle = None
        if self.color_object_angles:
            for angle in self.color_object_angles:
                print("angle", angle)
                ratio = math.pi * 2 / 640
                print("ratio", ratio)
                mode = abs(angle) // ratio
                print("mode", mode)
                if angle > 0.0:
                    dist = min(self.lidar[480+int(mode)-4:480+int(mode)+4])
                    if dist < min_dist:
                        min_dist = dist
                        cam_angle = angle
                    print(dist)
                else:
                    print("else")
                    dist = min(self.lidar[480 - int(mode) - 4:480 - int(mode) + 4])
                    if dist < min_dist:
                        min_dist = dist
                        cam_angle = angle
                    print(dist)
            #cam_angle = self.color_object_angles[0]


            print(f"Adjusting with camera. Δ{cam_angle:.4f} rad")

            # Kitsam piir - täpsem keskele asetamine
            if cam_angle > 0.02:
                self.left_velocity = 0.2
                self.right_velocity = -0.2
            elif cam_angle < -0.02:
                self.left_velocity = -0.2
                self.right_velocity = 0.2
            else:
                print("Object centered precisely. Driving toward target.")
                self.left_velocity = 2.5
                self.right_velocity = 2.5
                self.state = "driving"
        else:
            print("Object lost during adjustment. Switching to search.")
            self.state = "search"

    def _handle_driving(self):
        distance = self._get_front_distance()
        print(f"Driving to target. Distance: {distance:.2f}m")
        if distance < 0.4:
            print("Arrived at target! Waiting before next target...")
            self.left_velocity = 0
            self.right_velocity = 0
            self.arrival_time = self.robot.get_time()
            self.state = "waiting"

    def _handle_waiting(self):
        current_time = self.robot.get_time()
        if self.arrival_time is not None and current_time - self.arrival_time >= 5.0:
            self.state = "finished"

    def _handle_finished(self):
        print(f"FINISHED: {self.color_order[self.current_color_index]}")
        self.reset_detection_data()
        self._next_color()
        self.state = "search"
        self.search_timer = 0
        self.previous_time = self.robot.get_time()

    def reset_detection_data(self):
        self.detected_objects = []
        self.color_object_angles = []
        self.best_target_angle = None
        self.best_target_distance = float('inf')
        self.scan_start_angle = None
        self.arrival_time = None

    def _next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.color_order)

    def handle_no_colour(self):
        current_time = self.robot.get_time()
        timestep = current_time - self.previous_time
        self.previous_time = current_time

        if not self.color_object_angles:
            self.search_timer += timestep
            print(f"Looking for {self.current_color}... [{self.search_timer:.2f}s elapsed]")
            if self.search_timer > self.max_search_duration:
                print(f"Skipping {self.current_color} – not found in time")
                self._next_color()
                self.reset_detection_data()
                self.search_timer = 0
        else:
            self.search_timer = 0

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
