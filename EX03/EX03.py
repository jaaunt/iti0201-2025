"""EX03: Sensors."""
import math


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.detected_objects = []

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.range_list = self.robot.range_list

        if not self.range_list:
            self.detected_objects = []
            return

        objects = []
        in_object = False
        start_idx = None

        for i in range(1, len(self.range_list)):
            if self.range_list[i] is None or self.range_list[i] == float('inf') or self.range_list[i - 1] is None or \
                    self.range_list[i - 1] == float('inf'):
                in_object = False
                continue

            if not in_object and self.range_list[i] < self.range_list[i - 1] * 0.9:
                in_object = True
                start_idx = i

            elif in_object and self.range_list[i] > self.range_list[start_idx] * 1.1:
                center_idx = round(start_idx + (i - start_idx) / 2)
                objects.append((self.range_list[center_idx], self._get_angle(center_idx)))
                in_object = False

        self.detected_objects = self._filter_objects(objects)

    def get_objects_range_list(self) -> list | None:
        """Return the detected objects range list.

        Based on the robot's lidar range list measurements, extract objects and
        return a list of detected objects. Each object contains the distance
        in meters and angle in radians in terms of the scan.

        The expected angle is the angle for the index that is the center of the
        object (floored).

        Example:
            For example, object exists at lidar range list indexes 7, 8, 9, 10.
            Then the angle should be the same as it was for index 8. The
            distance should also be the same as it was for index 8.

        Returns:
            list: A list of tuples, where each tuple represents an object with
            distance and angle [(distance, angle), (distance, angle), ...].
            None if no objects are detected.
        """
        return self.detected_objects if self.detected_objects else None

    def _get_angle(self, index):
        num_points = len(self.range_list)
        fov = 2 * math.pi
        angle_per_step = fov / num_points
        return index * angle_per_step

    def _filter_objects(self, objects):
        min_distance_threshold = 0.5
        valid_objects = []

        for obj in objects:
            if obj[0] > min_distance_threshold:
                valid_objects.append(obj)

        return valid_objects

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
        print(self.get_objects_range_list())
