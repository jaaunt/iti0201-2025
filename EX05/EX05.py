"""EX05: Triangle Forming."""

from __future__ import annotations
import math


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

        self.robot_x = self.robot_y = self.theta = 0.0
        self.prev_left_ticks = self.prev_right_ticks = self.prev_time = 0
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
        if not (lidar := self.robot.get_lidar_range_list()):
            return None

        objects = []
        for i in range(1, len(lidar)):
            if float('inf') in (lidar[i], lidar[i - 1]):
                continue
            if abs(lidar[i] - lidar[i - 1]) > 0.1:
                objects.append((lidar[i], (i / len(lidar)) * (2 * np.pi)))

        if len(objects) < 2:
            return None

        (x1, y1), (x2, y2) = [(self.robot_x - d * math.sin(a), self.robot_y - d * math.cos(a)) for d, a in objects[:2]]
        dx, dy = (math.sqrt(3) / 2) * (y2 - y1), (math.sqrt(3) / 2) * (x2 - x1)
        return (x1 + x2) / 2 + dx, (y1 + y2) / 2 - dy, (x1 + x2) / 2 - dx, (y1 + y2) / 2 + dy

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Return the robot's pose as a tuple, based on wheel encoders and IMU.

        Returns:
            A tuple representing the (x, y, theta) robot's pose. Theta is the
            angle between robot's starting direction and its current direction
            (in radians, with -pi < theta <= pi).
        """
        if (dt := self.robot.get_time() - self.prev_time) <= 0:
            return self.robot_x, self.robot_y, self.theta

        left_ticks, right_ticks = self.robot.get_left_motor_encoder_ticks(), self.robot.get_right_motor_encoder_ticks()
        left_vel, right_vel = [(ticks - prev) / self.TICKS_PER_RADIANS / dt for ticks, prev in
                               zip((left_ticks, right_ticks), (self.prev_left_ticks, self.prev_right_ticks))]

        linear_vel = self.WHEEL_RADIUS * (left_vel + right_vel) / 2
        angular_vel = self.WHEEL_RADIUS * (right_vel - left_vel) / self.WHEEL_BASE

        self.theta = ((self.theta + angular_vel * dt + math.pi) % (2 * math.pi)) - math.pi
        self.robot_x += linear_vel * math.cos(self.theta) * dt
        self.robot_y += linear_vel * math.sin(self.theta) * dt

        self.prev_time, self.prev_left_ticks, self.prev_right_ticks = self.robot.get_time(), left_ticks, right_ticks
        return self.robot_x, self.robot_y, self.theta

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
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
