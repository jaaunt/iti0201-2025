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

        # Avoidance logic
        self.avoiding_obstacle = False
        self.avoid_start_time = 0.0
        self.avoid_duration = 1.2

        # After avoidance forward motion
        self.post_avoid_forward = False
        self.post_avoid_start = 0.0
        self.post_avoid_duration = 1.0

        # Blind push after cube disappears
        self.blind_push = False
        self.blind_push_start = 0.0
        self.blind_push_duration = 3.0
        self.was_adjusting = False

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

            if self.state == "search":
                print("Cube found")
        else:
            self.target_box = None

    def plan(self) -> None:
        current_time = self.robot.get_time()

        # LIDAR scan regions
        front = self.lidar[470:490] if self.lidar else []
        left = self.lidar[400:470] if self.lidar else []
        right = self.lidar[490:560] if self.lidar else []

        min_front = min((d for d in front if d), default=1.0)
        min_left = min((d for d in left if d), default=1.0)
        min_right = min((d for d in right if d), default=1.0)
        obstacle_close = min_front < 0.5 or min_left < 0.5 or min_right < 0.5

        # Track if we were adjusting
        if self.state == "adjusting":
            self.was_adjusting = True

        # End avoidance
        if self.avoiding_obstacle and current_time - self.avoid_start_time >= self.avoid_duration:
            print("Avoidance time ended, continuing straight")
            self.avoiding_obstacle = False
            self.post_avoid_forward = True
            self.post_avoid_start = current_time

        # Enter avoidance
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

        elif self.blind_push:
            if current_time - self.blind_push_start < self.blind_push_duration:
                self.state = "blind_push"
            else:
                print("Blind push complete")
                self.blind_push = False
                self.state = "arrived"

        elif self.target_box:
            if abs(self.target_angle) > 0.1:
                if self.state != "adjusting":
                    print("Adjusting to face cube")
                self.state = "adjusting"
            elif self.target_distance > 0.15:
                if self.state != "driving":
                    print("Driving toward cube")
                self.state = "driving"
            else:
                if self.state != "arrived":
                    print("Arrived at cube")
                self.state = "arrived"

        elif self.target_box is None and self.was_adjusting and not self.blind_push:
            print("Cube lost after adjustment. Starting blind push")
            self.blind_push = True
            self.blind_push_start = current_time
            self.was_adjusting = False
            self.state = "blind_push"

        elif current_time - self.last_seen_time > 10:
            if self.state != "search":
                print("Searching for cube")
            self.state = "search"

        # --- Movement control ---
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

        elif self.state == "arrived":
            self.left_velocity = 0
            self.right_velocity = 0

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    # --- Cube detection ---
    def get_cube_objects(self) -> list | None:
        if self.image is None:
            return None

        try:
            blue_channel = self.image[:, :, 0]
            green_channel = self.image[:, :, 1]
            red_channel = self.image[:, :, 2]
        except IndexError:
            return None

        threshold = 50
        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)

        labeled_mask, count = self.find_blobs(mask)
        if count == 0:
            return None

        boxes = []
        for i in range(1, count + 1):
            pixels = np.column_stack(np.where(labeled_mask == i))
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
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y, x in np.argwhere(mask):
            if labeled[y, x] == 0:
                labeled[y, x] = label_id
                to_visit.append((y, x))
                while to_visit:
                    cy, cx = to_visit.pop()
                    for dy, dx in neighbors:
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
