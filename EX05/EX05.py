"""EX05: Triangle Forming."""

from __future__ import annotations
import math
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.WHEEL_BASE = 0.233
        self.WHEEL_RADIUS = 0.03575
        self.TICKS_PER_RADIANS = 508.8 / (2 * math.pi)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.theta = 0.0

        self.prev_left_ticks = self.robot.get_left_motor_encoder_ticks()
        self.prev_right_ticks = self.robot.get_right_motor_encoder_ticks()

        self.prev_time = self.robot.get_time()

        self.lidar_data = self.robot.get_lidar_range_list()

        self.start_orientation = None

    def get_triangle_vertex_coordinates(self) -> tuple | None:
        """Return the triangle corner coordinates.

        Based on lidar range list and current robot position, calculate the world
        position of the equilateral triangle corner, and return coordinates of
        x, y.

        Logic:
        - This method uses lidar data to find the two objects that form the base of
          triangle (vertex)
        - Based on the found objects transform them to world frame coordinates and
          calculate triangle corner coordinates (there are two solutions since the
          equilateral triangle can be formed on both sides of the line connecting
          the two objects).
        - The robot's orientation and position are used to compute the actual world
          coordinates of the corner.

        Returns:
            A tuple of two tuples representing the (x, y) world coordinates of the
            two possible equilateral triangle's corners.
            (i.e., ((x1, y1), (x2, y2)))
            Returns `None` if no valid triangle corner can be detected.
        """
        self.detected_objects = []
        self.lidar_data = self.robot.get_lidar_range_list()
        if self.lidar_data is None:
            return None

        start_index = None
        threshold = 0.1
        object_size_min = 1

        for i in range(1, len(self.lidar_data)):
            if float('inf') in (self.lidar_data[i], self.lidar_data[i - 1]):
                start_index = None
                continue
            delta = self.lidar_data[i] - self.lidar_data[i - 1]

            if start_index is None and abs(delta) > threshold and delta < 0:
                start_index = i
            elif start_index is not None and abs(delta) > threshold and delta > 0:
                end_index = i - 1
                if abs(end_index - start_index) >= object_size_min:
                    object_values = self.lidar_data[start_index:end_index]
                    distance = np.min(object_values)
                    index = np.argmin(object_values)
                    center_index = start_index + index
                    angle = (center_index / len(self.lidar_data)) * (2 * np.pi)
                    self.detected_objects.append((distance, angle))
                start_index = None

        if len(self.detected_objects) < 2:
            return None

        obj_coords_robot = [(-d * math.sin(a), -d * math.cos(a)) for d, a in self.detected_objects[:2]]
        obj_coords_world = [(self.robot_x + x * math.cos(self.theta) - y * math.sin(self.theta),
                             self.robot_y + x * math.sin(self.theta) + y * math.cos(self.theta))
                            for x, y in obj_coords_robot]

        (x1, y1), (x2, y2) = obj_coords_world
        dx, dy = (math.sqrt(3) / 2) * (y2 - y1), (math.sqrt(3) / 2) * (x2 - x1)
        return ((x1 + x2) / 2 + dx, (y1 + y2) / 2 - dy), ((x1 + x2) / 2 - dx, (y1 + y2) / 2 + dy)

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Return the robot's pose as a tuple, based on wheel encoders and IMU.

        Returns:
            A tuple representing the (x, y, theta) robot's pose. Theta is the
            angle between robot's starting direction and its current direction
            (in radians, with -pi < theta <= pi).
        """
        curr_time = self.robot.get_time()
        dt = curr_time - self.prev_time
        if dt <= 0:
            return self.robot_x, self.robot_y, self.theta

        left_ticks = self.robot.get_left_motor_encoder_ticks()
        right_ticks = self.robot.get_right_motor_encoder_ticks()

        delta_left_ticks = left_ticks - self.prev_left_ticks
        delta_right_ticks = right_ticks - self.prev_right_ticks

        left_vel = (delta_left_ticks / self.TICKS_PER_RADIANS) / dt
        right_vel = (delta_right_ticks / self.TICKS_PER_RADIANS) / dt

        linear_vel = self.WHEEL_RADIUS * (left_vel + right_vel) / 2
        angular_vel = self.WHEEL_RADIUS * (right_vel - left_vel) / self.WHEEL_BASE

        self.theta = ((self.theta + angular_vel * dt + math.pi) % (2 * math.pi)) - math.pi
        self.robot_x += linear_vel * math.cos(self.theta) * dt
        self.robot_y += linear_vel * math.sin(self.theta) * dt

        self.prev_time = curr_time
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks

        return self.robot_x, self.robot_y, self.theta

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.lidar_data = self.robot.get_lidar_range_list()
        self.prev_time = self.robot.get_time()
        if self.start_orientation is None:
            self.start_orientation = self.robot.get_orientation()
        self.theta = self.robot.get_orientation() - self.start_orientation

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        pass

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        pass

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
