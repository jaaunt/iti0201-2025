from __future__ import annotations
import numpy as np
import math


def simplify_color_data_optimised(array, color):
    """Simplify to 0s and 1s: 1 if blue, else 0."""
    blue = array[:, :, 0].astype(float)
    green = array[:, :, 1].astype(float)
    red = array[:, :, 2].astype(float)

    if color == "BLUE":
        mask = (blue > 100) & ((green + red) < 143)
        return mask.astype(int)
    if color == "RED":
        mask = (red > 95) & (green < 60) & (blue < 60)
        return mask.astype(int)
    if color == "YELLOW":
        mask = (red > 110) & (green > 107) & (blue < 55)
        return mask.astype(int)


def clean_indexes(input_list):
    if input_list:
        result = []
        last_element = None
        for i in input_list:
            if not result:
                result.append(i + 1)
            elif i != last_element + 1:
                result.append(last_element - 1)
                result.append(i + 1)
            last_element = i
        result.append(last_element - 1)
        return result
    return None


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.orientation = None
        self.angle = None
        self.distance = None
        self.closest_object = None
        self.theta = None

        self.right_motor = 0.0
        self.left_motor = 0.0

        self.object_close = False
        self.current_data = None
        self.data = None

        self.x = 0.0
        self.y = 0.0
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0

        self.right_motor_velocity = 0.0
        self.left_motor_velocity = 0.0
        self.wheel_radius = self.robot.WHEEL_DIAMETER / 2
        self.wheel_base = 0.16
        self.ticks_per_revolution = 508.8
        self.linear_velocity = 0.0

        self.prev_time = self.robot.get_time()
        self.color_array = []
        self.fov = None
        self.spin_count = 1
        self.historic_y_coords = []
        self.historic_angles = []
        self.final_direction = None

    def update_odometry(self):
        self.theta = self.robot.get_orientation()

        current_left_ticks = self.robot.get_left_motor_encoder_ticks()
        current_right_ticks = self.robot.get_right_motor_encoder_ticks()

        left_delta = current_left_ticks - self.prev_left_ticks
        right_delta = current_right_ticks - self.prev_right_ticks

        left_dist = (left_delta / self.ticks_per_revolution) * (2 * math.pi * self.wheel_radius)
        right_dist = (right_delta / self.ticks_per_revolution) * (2 * math.pi * self.wheel_radius)
        center_dist = (left_dist + right_dist) / 2

        self.x += center_dist * math.cos(self.theta)
        self.y += center_dist * math.sin(self.theta)

        self.prev_left_ticks = current_left_ticks
        self.prev_right_ticks = current_right_ticks

        current_time = self.robot.get_time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        self.left_motor_velocity = left_dist / dt if dt > 0 else 0.0
        self.right_motor_velocity = right_dist / dt if dt > 0 else 0.0
        self.linear_velocity = (self.right_motor_velocity + self.left_motor_velocity) / 2

    def get_cube_bounding_box_list(self) -> list | None:
        if self.color_array is None:
            return None

        simplified = np.asarray(simplify_color_data_optimised(self.color_array, "BLUE"))
        column_height = len(simplified)
        blue_columns = [col for col in range(simplified.shape[1]) if np.any(simplified[:, col] == 1)]

        blue_indexes = clean_indexes(sorted(blue_columns))
        if not blue_indexes:
            return None

        result = []
        for n in range(0, len(blue_indexes), 2):
            x_min = blue_indexes[n]
            x_max = blue_indexes[n + 1]
            y_min = min(np.argmax(simplified[:, col] == 1) for col in range(x_min, x_max + 1))
            y_max = column_height - min(np.argmax(simplified[:, col][::-1] == 1) for col in range(x_min, x_max + 1)) - 1

            # Optional: Relax shape filter for testing
            if 0.5 * (x_max - x_min) <= (y_max - y_min) <= 1.2 * (x_max - x_min):
                result.append((x_min, x_max, y_min, y_max))

        return result if result else None

    def get_object_location_list(self) -> list | None:
        objects = self.get_cube_bounding_box_list()
        if not objects:
            return None

        detected = []
        cam_w = self.color_array.shape[1]
        cam_center_x = cam_w / 2

        for (x_min, x_max, y_min, y_max) in objects:
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            angle = (x_center - cam_center_x) / cam_center_x * (self.fov / 2)
            detected.append([x_center, y_center, angle])
        self.angle = detected[0][2] if detected else None
        return detected

    def sense(self) -> None:
        self.orientation = self.robot.get_orientation()
        self.color_array = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()

        self.update_odometry()

        objects = self.get_object_location_list()

        if objects:
            self.closest_object = objects[0]
            x_center, y_center, self.angle = self.closest_object

            if 630 < x_center < 650:
                self.distance = (900 - y_center) * 0.0061
                self.historic_y_coords.append(y_center)
                self.historic_angles.append(self.angle)
        elif len(self.historic_y_coords) > 3:
            if not self.final_direction:
                self.final_direction = self.robot.get_orientation()
            last = self.historic_y_coords[-1]
            second_last = self.historic_y_coords[-2]
            extrapolated = 2 * last - second_last

            self.historic_y_coords.append(extrapolated)
            self.distance = (900 - extrapolated) * 0.0061
            self.angle = self.robot.get_orientation() - self.final_direction
            self.closest_object = [1, 1, self.angle]
        else:
            print(f"Scanning... {len(self.historic_y_coords)}")

    def plan(self) -> None:
        if self.spin_count < 20:
            self.angle = self.robot.get_orientation() - 1.81
            self.distance = 2
            self.closest_object = [1, 1, self.angle]

        if self.linear_velocity <= 0:
            self.right_motor = 0.0
            self.left_motor = 0.0
        elif self.angle is None or self.distance is None or self.closest_object is None:
            self.right_motor = -0.05
            self.left_motor = 0.05
        elif self.distance < 0.4:
            self.right_motor = -0.03
            self.left_motor = -0.03
        elif -0.05 < self.angle - self.theta < 0.05:
            self.right_motor = 0.00105
            self.left_motor = 0.00105
        else:
            if self.angle > 0.0:
                self.right_motor = -0.06
                self.left_motor = 0.06
            else:
                self.right_motor = 0.06
                self.left_motor = -0.06

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_motor)
        self.robot.set_right_motor_velocity(self.right_motor)

    def spin(self) -> None:
        self.spin_count += 1
        self.sense()
        self.plan()
        self.act()
