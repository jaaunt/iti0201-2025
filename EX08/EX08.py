"""EX08: Mapping the Environment."""

class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.map = {}
        self.traversable_cells = set()
        self.unmapped_cells = set()
        self.grid_size = 0.615
        self.position = (0, 0)
        self.lidar_data = None

    def get_traversable_cells(self) -> list:
        """Get a list of all known traversable cells in the map.

        This method returns the grid cells that the robot knows to be traversable,
        starting from the initial position (0, 0). A traversable cell is a cell
        that the robot can go to.

        Returns:
            [(int, int), ...]: A list of tuples, where each tuple (x, y)
            represents the coordinates of a traversable cell.
        """
        return list(self.traversable_cells)

    def get_unmapped_cells(self) -> list:
        """Get a list of all unmapped cells that the robot has discovered so far.

        This method identifies grid cells that have been found but not yet fully mapped,
        starting with the initial position (0, 0).
        A cell is considered mapped when the robot has gathered a LIDAR reading while
        located in that cell. Then it can be removed from unmapped cells.
        This method returns a subset of all traversable cells.

        Returns:
            [(int, int), ...]: A list of tuples, where each tuple (x, y)
            represents the coordinates of an unmapped cell.
        """
        return list(self.unmapped_cells)

    def get_map(self) -> dict:
        """Get the map representation as a dictionary of adjacency.

        The map is represented as a dictionary where each key is a grid cell
        (represented as a tuple of coordinates), and the corresponding value
        is a list of adjacent cells.
        Adjacency is defined as orthogonal movement meaning: up, down, left, or right.

        Returns:
            {(int, int): [(int, int), ...]}: A dictionary where keys are cells (x, y)
            and values are lists of neighboring cells (x, y).
        """
        return self.map

    def update_map(self):
        pass

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.position = self.robot.get_current_position()
        self.lidar_data = self.robot.get_lidar_range_list()

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
