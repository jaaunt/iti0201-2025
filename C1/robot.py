from __future__ import annotations
import math
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.state = "init"
        self.left_velocity = 0
        self.right_velocity = 0

    def sense(self) -> None:
        """Gather sensor data from camera and LIDAR."""
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.blue_object_angles = self._get_blue_object_angles()

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
            x_center = (pixels[:, 1].min() + pixels[:, 1].max()) / 2
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

    def plan(self) -> None:
        """Plan the robot's actions."""
        state_actions = {
            "init": self._handle_init,
            "search": self._handle_search,
            "turning": self._handle_turning,
            "approaching": self._handle_approaching,
            "fixing_trajectory": self._handle_fixing_trajectory,
            "finished": self._handle_finished,
        }

        if self.state in state_actions:
            state_actions[self.state]()

    def _handle_init(self):
        print("HELLO, I ROBOT!")
        self.state = "search"

    def _handle_search(self):
        self.left_velocity = -5
        self.right_velocity = 5
        if self.blue_object_angles:
            self.state = "turning"

    def _handle_turning(self):
        if not self.blue_object_angles:
            self.state = "search"
            return

        target_angle = self.blue_object_angles[0]
        if abs(target_angle) > 0.1:
            self.left_velocity = -1 if target_angle > 0 else 1
            self.right_velocity = 1 if target_angle > 0 else -1
        else:
            self.state = "approaching"

    def _handle_approaching(self):
        self.left_velocity = 1
        self.right_velocity = 1
        if self.lidar and min(self.lidar) < 0.3:
            self.state = "finished"

    def _handle_fixing_trajectory(self):
        if not self.blue_object_angles:
            self.state = "search"
            return

        target_angle = self.blue_object_angles[0]
        if abs(target_angle) > 1:
            self.left_velocity = -0.1 if target_angle > 0 else 0.1
            self.right_velocity = 0.1 if target_angle > 0 else -0.1
        else:
            self.state = "approaching"

    def _handle_finished(self):
        self.left_velocity = 0
        self.right_velocity = 0
        print("I, END(myself)")

    def act(self) -> None:
        """Execute planned actions."""
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        """Main sense-plan-act loop."""
        self.sense()
        self.plan()
        self.act()
