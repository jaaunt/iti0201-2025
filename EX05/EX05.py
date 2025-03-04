from __future__ import annotations
from math import cos, sin, pi, sqrt


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.objects = []

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Returns:
            A tuple representing the (x, y, theta) robot's pose.
        """
        odometry = self.robot.get_odometry()
        self.x, self.y, self.theta = odometry['x'], odometry['y'], odometry['theta']
        return self.x, self.y, self.theta

    def sense(self) -> None:
        """Gather sensor data."""
        lidar_data = self.robot.get_lidar()
        self.objects = self.detect_objects(lidar_data)

    def detect_objects(self, lidar_data: list) -> list:
        """Detect cylindrical objects from lidar data."""
        objects = []
        angle_increment = self.robot.get_lidar_angle_increment()
        for i, distance in enumerate(lidar_data):
            if 0.1 < distance < 2.0:  # Assuming relevant object detection range
                angle = i * angle_increment - pi / 2
                x_r = distance * cos(angle)
                y_r = distance * sin(angle)
                x_w, y_w = self.transform_to_world(x_r, y_r)
                objects.append((x_w, y_w))
        return objects

    def transform_to_world(self, x_r: float, y_r: float) -> tuple:
        """Transform coordinates from robot frame to world frame."""
        x_w = self.x + x_r * cos(self.theta) - y_r * sin(self.theta)
        y_w = self.y + x_r * sin(self.theta) + y_r * cos(self.theta)
        return x_w, y_w

    def get_triangle_vertex_coordinates(self) -> tuple | None:
        """Return the triangle corner coordinates."""
        if len(self.objects) < 2:
            return None

        (x1, y1), (x2, y2) = self.objects[:2]
        side_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Compute midpoint
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        # Compute height of equilateral triangle
        height = (sqrt(3) / 2) * side_length

        # Compute possible vertices using perpendicular vector
        dx, dy = (y2 - y1), -(x2 - x1)
        norm = sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / norm, dy / norm

        vertex1 = (xm + height * dx, ym + height * dy)
        vertex2 = (xm - height * dx, ym - height * dy)

        return vertex1, vertex2

    def plan(self) -> None:
        """Plan the robot's actions."""
        self.get_robot_pose()

    def act(self) -> None:
        """Execute planned actions."""
        pass

    def spin(self) -> None:
        """Main loop for sense-plan-act cycle."""
        self.sense()
        self.plan()
        self.act()
