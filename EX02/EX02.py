"""EX02: State Machines."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.state = None

    def get_state(self) -> str:
        """Extract the current state of the robot based on Lidar sensor readings.

        Logic:
            - This function uses the Lidar range readings to determine the current state
            of the robot in terms of distance to obstacle directly in front of it.
            - The state is assigned based on the following conditions:
                - Very Far: If the range is greater than or equal to 1.5 meters.
                - Far: If the range is between (inclusive) 1.0 and 1.5 meters.
                - Near: If the range is between (inclusive) 0.7 and 1.0 meters.
                - Close: If the range is less than 0.7 meters.
            - If no new measurement is available from the Lidar sensor, the function
            will retain the previous state.

        Example:
            If the Lidar front facing reading returns a distance of 0.65 meters,
            the robot's state will be "close". If the distance is 1.2 meters, the state
            will be "far".

        Note:
            You will have to determine and extract the Lidars front facing direction
            measurement from the Lidar sensor
            readings.

        Returns:
            str: The current state of the robot based on its distance from an obstacle:
                 ("very far", "far", "near", or "close").
        """
        lidar_ranges = self.robot.get_lidar_range_list()  # saa see data
        if not lidar_ranges:  # kui no data
            return None

        front_distance = lidar_ranges[len(lidar_ranges) // 2]  # assumin et front facing vaartus umbes keskel

        if front_distance >= 1.5 or front_distance == "inf":
            return "very far"
        elif 1 <= front_distance < 1.5:
            return "far"
        elif 0.7 <= front_distance < 1:
            return "near"
        else:
            return "close"

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        if self.get_state():
            self.state = self.get_state()
        else:
            return self.state

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        Empty.
        """
        pass

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        Empty.
        """
        pass

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
