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

        self.scanning = False
        self.scan_start_orientation = None
        self.current_orientation = 0
        self.scan_completed = False
        self.best_box = None

        self.has_pending_target = False  # << NEW FLAG

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()

    def sense(self) -> None:
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.current_orientation = math.degrees(self.robot.get_orientation()) % 360

        boxes = self.get_cube_objects()
        if boxes:
            self.target_box = boxes[0]
            self.target_angle = self.calculate_angle(self.target_box)
            self.target_distance = self.estimate_distance(self.target_box)
            self.last_seen_time = self.robot.get_time()
            if self.state in ["search", "scanning"]:
                print("Cube found")
            if self.state == "scanning":
                self.best_box = self.target_box
        else:
            self.target_box = None

    def plan(self) -> None:
        current_time = self.robot.get_time()

        if self.state == "search" and not self.scanning:
            print("Starting 360 scan")
            self.scanning = True
            self.scan_start_orientation = self.current_orientation
            self.state = "scanning"
            self.best_box = None

        elif self.state == "scanning":
            angle_diff = (self.current_orientation - self.scan_start_orientation + 360) % 360
            if angle_diff >= 360:
                self.scanning = False
                if self.best_box:
                    print("Cube found during scan")
                    self.target_box = self.best_box
                    self.target_angle = self.calculate_angle(self.target_box)
                    self.target_distance = self.estimate_distance(self.target_box)
                    self.has_pending_target = True  # << remember cube even if it disappears
                    self.state = "adjusting"
                else:
                    print("Scan complete – cube not found")
                    self.state = "done"
            else:
                self.left_velocity = -0.5
                self.right_velocity = 0.5
                return

        elif self.target_box or self.has_pending_target:
            if abs(self.target_angle) > 0.1:
                print("Adjusting to face cube")
                self.state = "adjusting"
            elif self.target_distance > 0.15:
                print("Driving toward cube")
                self.state = "driving"
            else:
                print("Arrived at cube")
                self.state = "arrived"
                self.has_pending_target = False

        elif current_time - self.last_seen_time > 10:
            print("Searching for cube")
            self.state = "search"

        if self.state == "adjusting":
            self.left_velocity = 0.3 if self.target_angle > 0 else -0.3
            self.right_velocity = -self.left_velocity

        elif self.state == "driving":
            self.left_velocity = 1.5
            self.right_velocity = 1.5

        elif self.state == "done":
            self.left_velocity = 0
            self.right_velocity = 0

        elif self.state == "arrived":
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
