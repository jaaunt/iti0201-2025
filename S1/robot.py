"""S1."""
import math


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.detected_objects = []
        self.robot = robot
        self.state = "init"

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()

        # EX03 stuff
        if self.lidar is None:
            print("Lidar data is NULL!")
            self.range_list = []
            return
        else:
            self.range_list = self.lidar

        if not self.range_list or not isinstance(self.range_list, list):
            print("Invalid or empty Lidar data, skipping sensing.")
            self.range_list = []
            return

        objects = []
        in_object = False
        start_idx = None

        for i in range(1, len(self.range_list)):
            if self.range_list[i] is None or self.range_list[i] == float('inf') or self.range_list[i - 1] is None or \
                    self.range_list[i - 1] == float('inf'):
                in_object = False
                continue

            if not in_object and self.range_list[i] < self.range_list[i - 1] * 0.8:
                in_object = True
                start_idx = i

            elif in_object and self.range_list[i] > self.range_list[start_idx] * 1.2:
                center_idx = round(start_idx + (i - start_idx) / 2)
                objects.append((self.range_list[center_idx], self._get_angle(center_idx)))
                in_object = False

        self.detected_objects = self._filter_objects(objects)

    #EX03 methods
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
        if self.state == "init":
            print("HELLO, I ROBOT!")
        elif self.state == "search":
            self.left_velocity = 0.1
            self.right_velocity = -0.1
            if len(self.detected_objects) > 0:
                self.state = "turning_to_object"
        elif self.state == "turning_to_object":
            self.left_velocity = 0.1
            self.right_velocity = -0.1
            if self.detected_objects and len(self.detected_objects) > 0:
                if self.detected_objects[0][1] == 0: # peab olema range sest muidu ei saa aru
                    self.state = "approaching"
            else:
                self.state = "searching"

        elif self.state == "approaching":
            self.left_velocity = 0.1
            self.right_velocity = 0.1
            if len(self.detected_objects) > 0:
                if self.detected_objects[0][0] < 0.3: # peab olema range sest muidu ei saa aru
                    self.state = "finished"
            else:
                self.state = "searching"
        elif self.state == "finished":
            self.left_velocity = 0
            self.right_velocity = 0
            pass


    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)


    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
