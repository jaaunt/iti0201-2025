from __future__ import annotations
import math
import numpy as np


class Robot:
    """Turtlebot robot solving C3 task: approach blue cube, avoid obstacles."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.image = None
        self.fov = None
        self.lidar = None
        self.state = "search"
        self.start_angle = None
        self.best_box = None
        self.state_start_time = None
        self.left_velocity = 0.0
        self.right_velocity = 0.0

    def sense(self) -> None:
        print("Sensing environment...")
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()

    def plan(self) -> None:
        print(f"Current state: {self.state}")
        if self.state == "search":
            self._handle_search()
        elif self.state == "approach":
            self._handle_approach()
        elif self.state == "avoid_obstacle":
            self._handle_avoid_obstacle()
        elif self.state == "drive":
            self._handle_drive()
        elif self.state == "finished":
            print("Finished task.")
            self.left_velocity = 0
            self.right_velocity = 0

    def act(self) -> None:
        print(f"Acting: Left Velocity = {self.left_velocity}, Right Velocity = {self.right_velocity}")
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()

    def _handle_search(self):
        print("Searching for blue cube...")
        boxes = self._get_blue_cubes()
        if boxes:
            print(f"Found {len(boxes)} blue object(s). Evaluating...")
            self.best_box = min(boxes, key=lambda b: self._estimate_distance(b))
            print(f"Best box selected: {self.best_box}")
            self.state = "approach"
        else:
            print("No blue cube found. Rotating...")
            self.left_velocity = -1.0
            self.right_velocity = 1.0

    def _handle_approach(self):
        if self.best_box is None:
            print("Lost track of cube. Returning to search.")
            self.state = "search"
            return
        angle = self._bounding_box_angle(self.best_box)
        print(f"Approaching cube at angle {angle:.2f} radians")
        if self._is_obstacle_in_path(angle):
            print("Obstacle detected in path! Switching to avoidance.")
            self.state = "avoid_obstacle"
            self.state_start_time = self.robot.get_time()
        else:
            print("Path clear. Proceeding to drive toward cube.")
            self.state = "drive"

    def _handle_avoid_obstacle(self):
        direction = self._choose_clear_side()
        print(f"Avoiding obstacle. Chosen direction: {direction}")
        t = self.robot.get_time()
        if self.state_start_time is None:
            self.state_start_time = t

        elapsed = t - self.state_start_time
        if elapsed < 1.5:
            if direction == "left":
                self.left_velocity = -1.0
                self.right_velocity = 1.0
            else:
                self.left_velocity = 1.0
                self.right_velocity = -1.0
        elif elapsed < 3.0:
            print("Driving forward to clear obstacle...")
            self.left_velocity = 2.0
            self.right_velocity = 2.0
        else:
            print("Retrying search after avoidance maneuver.")
            self.state = "search"
            self.best_box = None
            self.state_start_time = None

    def _handle_drive(self):
        if self.best_box is None:
            print("No cube to drive to. Searching again.")
            self.state = "search"
            return

        angle = self._bounding_box_angle(self.best_box)
        dist = self._estimate_distance(self.best_box)
        print(f"Driving toward cube. Angle: {angle:.2f}, Distance: {dist:.2f}")

        if angle > 0.05:
            self.left_velocity = 0.3
            self.right_velocity = -0.3
        elif angle < -0.05:
            self.left_velocity = -0.3
            self.right_velocity = 0.3
        elif dist > 0.4:
            self.left_velocity = 2.0
            self.right_velocity = 2.0
        else:
            print("Arrived at cube!")
            self.left_velocity = 0.0
            self.right_velocity = 0.0
            self.state = "finished"

    def _get_blue_cubes(self):
        if self.image is None:
            return []

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        mask = (blue > green + 50) & (blue > red + 50)
        label_mask, count = self._find_blobs(mask)

        boxes = []
        for i in range(1, count + 1):
            pixels = np.column_stack(np.where(label_mask == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)
            if abs((x_max - x_min) - (y_max - y_min)) < 20:
                boxes.append((x_min, x_max, y_min, y_max))

        print(f"Detected {len(boxes)} potential blue cube(s).")
        return boxes

    def _find_blobs(self, mask):
        height, width = mask.shape
        label_mask = np.zeros_like(mask, dtype=np.uint32)
        label_id = 1
        to_visit = []
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for y, x in np.argwhere(mask):
            if label_mask[y, x] == 0:
                label_mask[y, x] = label_id
                to_visit.append((y, x))
                while to_visit:
                    cy, cx = to_visit.pop()
                    for dy, dx in neighbours:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] and label_mask[ny, nx] == 0:
                                label_mask[ny, nx] = label_id
                                to_visit.append((ny, nx))
                label_id += 1

        return label_mask, label_id - 1

    def _bounding_box_angle(self, box):
        x_min, x_max, _, _ = box
        x_center = (x_min + x_max) / 2
        width = self.image.shape[1]
        angle = ((x_center - width / 2) / (width / 2)) * (self.fov / 2)
        return angle

    def _estimate_distance(self, box):
        _, _, y_min, y_max = box
        height = y_max - y_min
        return 1.0 / (height + 1e-6) * 30

    def _is_obstacle_in_path(self, target_angle):
        center_index = 480 + int(target_angle / (self.fov / 640))
        check_range = self.lidar[max(0, center_index - 5):min(640, center_index + 5)]
        return any(d is not None and d < 0.6 for d in check_range)

    def _choose_clear_side(self):
        left = self.lidar[0:320]
        right = self.lidar[320:640]
        left_clear = np.nanmean([d for d in left if d and d != float('inf')])
        right_clear = np.nanmean([d for d in right if d and d != float('inf')])
        print(f"Clearance - Left: {left_clear:.2f}, Right: {right_clear:.2f}")
        return "left" if left_clear > right_clear else "right"
