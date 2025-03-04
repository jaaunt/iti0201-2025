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

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Return the robot's pose as a tuple, based on wheel encoders and IMU.

        Returns:
            A tuple representing the (x, y, theta) robot's pose. Theta is the
            angle between robot's starting direction and its current direction
            (in radians, with -pi < theta <= pi).
        """
        odometry = self.robot.get_odometry()
        self.x, self.y, self.theta = odometry['x'], odometry['y'], odometry['theta']
        return self.x, self.y, self.theta

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
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

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        self.get_robot_pose()

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
