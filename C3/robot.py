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

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()

    def sense(self) -> None:
        """Gather sensor data."""
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()

        print(f"[SENSE] Image: {'OK' if self.image is not None else 'None'}, FOV: {self.fov}, LIDAR len: {len(self.lidar) if self.lidar else 'None'}")

        boxes = self.get_cube_objects()
        if boxes:
            self.target_box = boxes[0]
            self.target_angle = self.calculate_angle(self.target_box)
            self.target_distance = self.estimate_distance(self.target_box)
            self.last_seen_time = self.robot.get_time()
            print(f"[SENSE] Cube found: box={self.target_box}, angle={self.target_angle:.2f}, distance={self.target_distance:.2f}")
        else:
            print("[SENSE] No cube found")
            self.target_box = None

    def plan(self) -> None:
        """Decide robot's next action."""
        current_time = self.robot.get_time()
        print(f"[PLAN] State: {self.state}")

        if self.target_box:
            if abs(self.target_angle) > 0.1:
                self.state = "adjusting"
            elif self.target_distance > 0.25:
                self.state = "driving"
            else:
                self.state = "arrived"
        elif current_time - self.last_seen_time > 10:
            self.state = "search"

        if self.state == "adjusting":
            self.left_velocity = 0.3 if self.target_angle > 0 else -0.3
            self.right_velocity = -self.left_velocity
            print(f"[PLAN] Adjusting: angle={self.target_angle:.2f}")

        elif self.state == "driving":
            front_lidar = self.lidar[470:490] if self.lidar else []
            if any(d < 0.3 for d in front_lidar if d):
                self.left_velocity = 0
                self.right_velocity = 0
                print("[PLAN] Obstacle detected! Stopping.")
            else:
                self.left_velocity = 1.5
                self.right_velocity = 1.5
                print(f"[PLAN] Driving toward cube. Distance: {self.target_distance:.2f}")

        elif self.state == "search":
            self.left_velocity = -0.5
            self.right_velocity = 0.5
            print("[PLAN] Searching for cube...")

        elif self.state == "arrived":
            self.left_velocity = 0
            self.right_velocity = 0
            print("[PLAN] Arrived at cube!")

    def act(self) -> None:
        print(f"[ACT] Left velocity: {self.left_velocity:.2f}, Right velocity: {self.right_velocity:.2f}")
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    # --- Image processing methods ---

    def get_cube_objects(self) -> list | None:
        if self.image is None:
            return None

        try:
            blue_channel = self.image[:, :, 0]
            green_channel = self.image[:, :, 1]
            red_channel = self.image[:, :, 2]
        except IndexError:
            print("[IMAGE] Invalid image format")
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
            if abs(x_len - y_len) <= 20:  # Cube-like
                boxes.append((x_min, x_max, y_min, y_max))
                print(f"[IMAGE] Detected cube-like object: {x_min}-{x_max}, {y_min}-{y_max}")

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
