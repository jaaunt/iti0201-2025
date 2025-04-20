"""EX09."""
import math
from collections import deque


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.traversable_cells = [(0, 0)]
        self.unmapped_cells = []
        self.map = {}
        self.lidar = None
        self.orientation = None
        self.current_position = None
        self.frontier = None

    def get_traversable_cells(self) -> list:
        """Get a list of all known traversable cells in the map.

        This method returns the grid cells that the robot knows to be traversable,
        starting from the initial position (0, 0). A traversable cell is a cell
        that the robot can go to.

        Returns:
            [(int, int), ...]: A list of tuples, where each tuple (x, y)
            represents the coordinates of a traversable cell.
        """
        return self.traversable_cells

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
        return self.unmapped_cells

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

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.orientation = self.robot.get_orientation()
        self.lidar_readings = self.robot.get_lidar_range_list()
        self.current_position = self.robot.get_current_position()
        if self.lidar_readings:
            self.front = self.lidar_readings[480]
            self.back = self.lidar_readings[150]
            self.right = self.lidar_readings[1]
            self.left = self.lidar_readings[320]

    def add_cells(self, cell, direction):
        """Add cells in the given direction from the current position."""
        x, y = self.current_position

        # Direction mapping
        directions = {
            "up": (0, 1),
            "down": (0, -1),  # same as "back"
            "back": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        if direction not in directions:
            return  # invalid direction

        dx, dy = directions[direction]

        for c in range(1, cell + 1):
            coord = (x + dx * c, y + dy * c)

            if c == 1:
                # Link current_position <-> new coord
                self.map.setdefault(self.current_position, []).append(coord)
                self.map.setdefault(coord, []).append(self.current_position)

            if coord not in self.traversable_cells:
                self.traversable_cells.append(coord)
                self.unmapped_cells.append(coord)

    def case1(self):
        """Map."""
        if self.front > 0.45:
            cell = self.front // 0.625
            self.add_cells(int(cell), "up")
        if self.back > 0.45:
            cell = self.back // 0.625
            self.add_cells(int(cell), "back")
        if self.right > 0.45:
            cell = self.right // 0.625
            self.add_cells(int(cell), "right")
        if self.left > 0.45:
            cell = self.left // 0.625
            self.add_cells(int(cell), "left")

    def case2(self):
        """Map."""
        if self.front > 0.45:
            cell = self.front // 0.625
            self.add_cells(int(cell), "left")
        if self.back > 0.45:
            cell = self.back // 0.625
            self.add_cells(int(cell), "right")
        if self.right > 0.45:
            cell = self.right // 0.625
            self.add_cells(int(cell), "up")
        if self.left > 0.45:
            cell = self.left // 0.625
            self.add_cells(int(cell), "back")

    def case3(self):
        """Map."""
        if self.front > 0.45:
            cell = self.front // 0.625
            self.add_cells(int(cell), "right")
        if self.back > 0.45:
            cell = self.back // 0.625
            self.add_cells(int(cell), "left")
        if self.right > 0.45:
            cell = self.right // 0.625
            self.add_cells(int(cell), "back")
        if self.left > 0.45:
            cell = self.left // 0.625
            self.add_cells(int(cell), "up")

    def mapping(self):
        """Map."""
        if -0.1 < self.orientation < 0.1:
            self.case1()
        if 1.47 < self.orientation < 1.67:
            self.case2()
        if -1.67 < self.orientation < -1.47:
            self.case3()

        if self.orientation > (math.pi - 0.1) or self.orientation < (-math.pi + 0.1):
            if self.front > 0.45:
                cell = self.front // 0.625
                self.add_cells(int(cell), "back")
            if self.back > 0.45:
                cell = self.back // 0.625
                self.add_cells(int(cell), "up")
            if self.right > 0.45:
                cell = self.right // 0.625
                self.add_cells(int(cell), "left")
            if self.left > 0.45:
                cell = self.left // 0.625
                self.add_cells(int(cell), "right")

    def get_frontier_and_path(self) -> list:
        """Identify next frontier for exploration and calculate the path to reach it.

        The frontier is the boundary between the known (mapped) and unknown (unmapped)
        regions of the map. This method determines the most suitable frontier to
        explore next and computes the path from the robot's current position to that
        frontier. Formula for choosing the next frontier to explore: Manhattan distance

        Returns:
            [(int, int), [(int, int), ...]]:
            - The first element is a tuple (x, y) representing the coordinates of the
              selected frontier.
            - The second element is a list of tuples [(x1, y1), (x2, y2), ...]
              representing the sequence of grid cells (coordinates) the robot should
              traverse in order to reach the frontier.

        Example:
            Suppose the robot's current position is (0, 0), and it detects a frontier
            at (3, 0). The function might return:
                [(3, 0), [(0, 0), (1, 0), (2, 0), (3, 0)]]
            This means the robot should travel through the listed cells to reach the
            frontier at (3, 0).
        """
        return self.frontier

    def find_frontier(self):
        """Find frontier."""
        print(self.unmapped_cells)
        if not self.unmapped_cells:
            return

        min_distance = float('inf')
        closest_cell = None

        # 1. Leia lÃ¤him unmapped cell (Manhattan distance)
        for cell in self.unmapped_cells:
            distance = abs(cell[0] - self.current_position[0]) + abs(cell[1] - self.current_position[1])
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell
        print(closest_cell)

        # 2. Leia tee sinna kasutades BFS

        def bfs(start, goal):
            queue = deque()
            queue.append((start, [start]))
            visited = set()
            visited.add(start)

            while queue:
                current, path = queue.popleft()
                if current == goal:
                    return path

                for neighbor in self.map.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            return []

        path = bfs(self.current_position, closest_cell)
        print(path)
        self.frontier = (closest_cell, path)
        if closest_cell in self.unmapped_cells:
            self.unmapped_cells.remove(closest_cell)

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        self.mapping()
        self.find_frontier()

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
