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

        self.lidar_range_list = []
        self.time = 0
        self.orientation = 0
        self.lidar_point_cloud = []
        self.left_encoder_ticks = 0
        self.right_encoder_ticks = 0
        self.robot_pose = (0, 0, 0)
        self.triangle_vertices = None
        self.vertices = ()

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
        if not self.lidar_point_cloud or len(self.lidar_point_cloud) < 2:
            return None

        # kaks laimat
        sorted_objects = sorted(self.lidar_point_cloud, key=lambda p: math.dist((0, 0), p))[:2]
        (x1_r, y1_r), (x2_r, y2_r) = sorted_objects

        x_w, y_w, theta = self.get_robot_pose()

        # need imelikud antud tehted
        x1_w = x_w + x1_r * math.cos(theta) - y1_r * math.sin(theta)
        y1_w = y_w + x1_r * math.sin(theta) + y1_r * math.cos(theta)

        x2_w = x_w + x2_r * math.cos(theta) - y2_r * math.sin(theta)
        y2_w = y_w + x2_r * math.sin(theta) + y2_r * math.cos(theta)

        # keskkoht
        mid_x, mid_y = (x1_w + x2_w) / 2, (y1_w + y2_w) / 2

        # korgus
        base_length = math.dist((x1_w, y1_w), (x2_w, y2_w))
        height = (math.sqrt(3) / 2) * base_length

        dx, dy = x2_w - x1_w, y2_w - y1_w
        perp_x, perp_y = -dy, dx
        norm = math.sqrt(perp_x ** 2 + perp_y ** 2)

        if norm == 0:
            return None  # ara jaga nulliga

        perp_x /= norm
        perp_y /= norm

        # leia molemad voimalikud voib olla all voi ulevaö
        x3_w_1 = mid_x + height * perp_x
        y3_w_1 = mid_y + height * perp_y

        x3_w_2 = mid_x - height * perp_x
        y3_w_2 = mid_y - height * perp_y

        return ((x3_w_1, y3_w_1), (x3_w_2, y3_w_2))

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Return the robot's pose as a tuple, based on wheel encoders and IMU.

        Returns:
            A tuple representing the (x, y, theta) robot's pose. Theta is the
            angle between robot's starting direction and its current direction
            (in radians, with -pi < theta <= pi).
        """
        return self.robot_pose

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.lidar_range_list = self.robot.get_lidar_range_list()
        self.time = self.robot.get_time()
        self.orientation = self.robot.get_orientation()
        self.lidar_point_cloud = self.robot.get_lidar_point_cloud()
        self.left_encoder_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_encoder_ticks = self.robot.get_right_motor_encoder_ticks()

        self.get_robot_pose()
        self.vertices = self.get_triangle_vertex_coordinates()

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
