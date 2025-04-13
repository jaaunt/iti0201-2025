from __future__ import annotations
import numpy as np
import math


class Robot:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.state = "search"
        self.image = None
        self.fov = None
        self.lidar = None
        self.target_box = None
        self.last_seen_time = 0.0
        self.target_angle = None
        self.target_distance = None
        self.left_velocity = 0
        self.right_velocity = 0

        self.avoiding_obstacle = False
        self.avoid_start_time = 0.0
        self.avoid_duration = 1.2

        self.post_avoid_forward = False
        self.post_avoid_start = 0.0
        self.post_avoid_duration = 1.0

        self.blind_push = False
        self.blind_push_start = 0.0
        self.blind_push_duration = 4.0
        self.was_adjusting = False

        self.scanning = False
        self.scan_start_angle = None
        self.rotation_threshold = 360
        self.scan_best_target = None
        self.scan_best_angle = None

        self.last_target_box_seen = False
        self.last_state = "search"

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()

    def sense(self) -> None:
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()

        boxes = self.get_cube_objects()
        if boxes:
            self.target_box = boxes[0]
            self.target_angle = self.calculate_angle(self.target_box)
            self.target_distance = self.estimate_distance(self.target_box)
            self.last_seen_time = self.robot.get_time()
            if self.state in ["search", "scanning"]:
                print("Cube found")
            if self.state == "scanning":
                self.scan_best_target = self.target_box
                self.scan_best_angle = math.degrees(self.robot.get_orientation()) % 360
        else:
            self.target_box = None

        self.last_target_box_seen = self.target_box is not None

    def plan(self) -> None:
        current_time = self.robot.get_time()
        self.last_state = self.state

        orientation = math.degrees(self.robot.get_orientation()) % 360

        front = self.lidar[470:490] if self.lidar else []
        left = self.lidar[400:470] if self.lidar else []
        right = self.lidar[490:560] if self.lidar else []
        min_front = min((d for d in front if d), default=1.0)
        min_left = min((d for d in left if d), default=1.0)
        min_right = min((d for d in right if d), default=1.0)
        obstacle_close = min_front < 0.5 or min_left < 0.5 or min_right < 0.5

        if self.avoiding_obstacle and current_time - self.avoid_start_time >= self.avoid_duration:
            print("Avoidance time ended, continuing straight")
            self.avoiding_obstacle = False
            self.post_avoid_forward = True
            self.post_avoid_start = current_time

        if obstacle_close and not self.avoiding_obstacle:
            print("Obstacle detected, entering avoidance mode")
            self.avoiding_obstacle = True
            self.avoid_start_time = current_time
            self.state = "avoiding"

        if self.avoiding_obstacle:
            self.state = "avoiding"

        elif self.post_avoid_forward:
            if current_time - self.post_avoid_start < self.post_avoid_duration:
                print("Post-avoid: moving straight")
                self.state = "post_forward"
            else:
                self.post_avoid_forward = False
                if not self.target_box and self.last_state in ["driving", "adjusting"]:
                    print("Cube likely close, starting blind push")
                    self.blind_push = True
                    self.blind_push_start = current_time
                    self.state = "blind_push"

        elif self.blind_push:
            if current_time - self.blind_push_start < self.blind_push_duration:
                print("Blind push: driving forward")
                self.state = "blind_push"
            else:
                print("Blind push complete – stopping permanently")
                self.blind_push = False
                self.state = "done"

        elif self.state == "search" and self.last_seen_time == 0.0 and not self.scanning:
            print("Starting scan – rotating to find cube")
            self.scanning = True
            self.scan_start_angle = orientation
            self.scan_best_target = None
            self.scan_best_angle = None
            self.state = "scanning"

        elif self.state == "scanning":
            angle_diff = (orientation - self.scan_start_angle + 360) % 360
            if angle_diff >= self.rotation_threshold:
                if self.scan_best_target is not None:
                    print("Scan complete – cube found during scan")
                    self.target_box = self.scan_best_target
                    self.target_angle = self.calculate_angle(self.target_box)
                    self.target_distance = self.estimate_distance(self.target_box)
                    self.state = "adjusting"
                else:
                    print("Scan complete – cube not found")
                    self.state = "done"
                self.scanning = False
            else:
                self.left_velocity = -0.5
                self.right_velocity = 0.5
                return

        elif self.target_box:
            if abs(self.target_angle) > 0.1:
                print("Adjusting to face cube")
                self.state = "adjusting"
            elif self.target_distance > 0.15:
                print("Driving toward cube")
                self.state = "driving"
            else:
                print("Arrived at cube")
                self.state = "arrived"

        elif not self.last_target_box_seen and not self.blind_push and self.last_state in ["adjusting", "driving"]:
            print("Cube lost after tracking. Starting blind push")
            self.blind_push = True
            self.blind_push_start = current_time
            self.state = "blind_push"

        elif current_time - self.last_seen_time > 10:
            print("Searching for cube")
            self.state = "search"

        if self.state == "adjusting":
            self.left_velocity = 0.3 if self.target_angle > 0 else -0.3
            self.right_velocity = -self.left_velocity

        elif self.state == "driving":
            self.left_velocity = 1.5
            self.right_velocity = 1.5

        elif self.state == "avoiding":
            print("Avoiding: rotating and moving forward")
            if min_left < min_right:
                self.left_velocity = 1.0
                self.right_velocity = 0.3
            else:
                self.left_velocity = 0.3
                self.right_velocity = 1.0

        elif self.state == "post_forward":
            self.left_velocity = 1.2
            self.right_velocity = 1.2

        elif self.state == "blind_push":
            self.left_velocity = 1.2
            self.right_velocity = 1.2

        elif self.state == "search":
            self.left_velocity = -0.5
            self.right_velocity = 0.5

        elif self.state == "arrived" or self.state == "done":
            self.left_velocity = 0
            self.right_velocity = 0

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)
        if self.state == "done":
            print("Act: Robot stopped permanently.")

    def get_cube_objects(self) -> list | None:
        if self.image is None:
            return None
        try:
            blue = self.image[:, :, 0]
            green = self.image[:, :, 1]
            red = self.image[:, :, 2]
        except IndexError:
            return None
        mask = (blue > green + 50) & (blue > red + 50)
        labeled, count = self.find_blobs(mask)
        if count == 0:
            return None
        boxes = []
        for i in range(1, count + 1):
            pixels = np.column_stack(np.where(labeled == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)
            x_len = x_max - x_min
            y_len = y_max - y_min
            if abs(x_len - y_len) <= 20:
                boxes.append((x_min, x_max, y_min, y_max))
        return boxes if boxes else None

    def find_blobs(self, mask):
        height, width = mask.shape
        labeled = np.zeros_like(mask, dtype=np.uint32)
        label_id = 1
        to_visit = []
        for y, x in np.argwhere(mask):
            if labeled[y, x] == 0:
                labeled[y, x] = label_id
                to_visit.append((y, x))
                while to_visit:
                    cy, cx = to_visit.pop()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = label_id
                                to_visit.append((ny, nx))
                label_id += 1
        return labeled, label_id - 1

    def calculate_angle(self, box):
        x_min, x_max, _, _ = box
        width = self.image.shape[1]
        x_center = (x_min + x_max) / 2
        angle = ((x_center - width / 2) / (width / 2)) * (self.fov / 2)
        return angle

    def estimate_distance(self, box):
        _, _, y_min, y_max = box
        height = y_max - y_min
        if height == 0:
            return float('inf')
        known_height_px = 80
        known_distance = 0.3
        return (known_height_px / height) * known_distance
