"""EX09: Mapping the Environment."""
import math
from queue import PriorityQueue


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
        self.current_position = None  # current (x, y) grid position
        self.frontier = None   # the target unmapped cell and the path to it

    def get_traversable_cells(self) -> list:
        """Return known traversable cells."""
        return self.traversable_cells

    def get_unmapped_cells(self) -> list:
        """Return cells discovered but not yet mapped with LIDAR."""
        return self.unmapped_cells

    def get_map(self) -> dict:
        """Return the map as a dictionary of cell adjacencies."""
        return self.map

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.orientation = self.robot.get_orientation()
        self.lidar = self.robot.get_lidar_range_list()
        self.current_position = self.robot.get_current_position()
        if self.lidar:
            self.front = self.lidar[480]  # front (0 degrees)
            self.back = self.lidar[150]  # back (180 degrees)
            self.right = self.lidar[1]  # right (270 degrees)
            self.left = self.lidar[320]  # left (90 degrees)

    def add_cells(self, cell, direction):
        """Add traversable cells in a given direction from current position."""
        x, y = self.current_position  # start from current position
        directions = {  # possible directions and how the values would have to change for them
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        if direction not in directions:
            return

        dx, dy = directions[direction]

        for c in range(1, cell + 1):
            coordinates = (x + dx * c, y + dy * c)

            if c == 1:
                self.map.setdefault(self.current_position, []).append(coordinates)
                self.map.setdefault(coordinates, []).append(self.current_position)

            if coordinates not in self.traversable_cells:
                self.traversable_cells.append(coordinates)
                self.unmapped_cells.append(coordinates)

    def facing_north(self):
        """Map surroundings assuming robot is facing north (0 rad)."""
        if self.front > 0.45:
            self.add_cells(int(self.front // 0.625), "up")
        if self.back > 0.45:
            self.add_cells(int(self.back // 0.625), "down")
        if self.right > 0.45:
            self.add_cells(int(self.right // 0.625), "right")
        if self.left > 0.45:
            self.add_cells(int(self.left // 0.625), "left")

    def facing_west(self):
        """Map surroundings assuming robot is facing west (π/2 rad)."""
        if self.front > 0.45:
            self.add_cells(int(self.front // 0.625), "left")
        if self.back > 0.45:
            self.add_cells(int(self.back // 0.625), "right")
        if self.right > 0.45:
            self.add_cells(int(self.right // 0.625), "up")
        if self.left > 0.45:
            self.add_cells(int(self.left // 0.625), "down")

    def facing_east(self):
        """Map surroundings assuming robot is facing east (-π/2 rad)."""
        if self.front > 0.45:
            self.add_cells(int(self.front // 0.625), "right")
        if self.back > 0.45:
            self.add_cells(int(self.back // 0.625), "left")
        if self.right > 0.45:
            self.add_cells(int(self.right // 0.625), "down")
        if self.left > 0.45:
            self.add_cells(int(self.left // 0.625), "up")

    def facing_south(self):
        """Map surroundings assuming robot is facing south (π rad)."""
        if self.front > 0.45:
            self.add_cells(int(self.front // 0.625), "down")
        if self.back > 0.45:
            self.add_cells(int(self.back // 0.625), "up")
        if self.right > 0.45:
            self.add_cells(int(self.right // 0.625), "left")
        if self.left > 0.45:
            self.add_cells(int(self.left // 0.625), "right")

    def mapping(self):
        """Determine orientation and map based on direction the robot is facing."""
        if -0.1 < self.orientation < 0.1:
            self.facing_north()
        elif 1.47 < self.orientation < 1.67:
            self.facing_west()
        elif -1.67 < self.orientation < -1.47:
            self.facing_east()
        elif self.orientation > (math.pi - 0.1) or self.orientation < (-math.pi + 0.1):
            self.facing_south()

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
        """Find the closest unmapped cell and compute a path using A*."""
        if not self.unmapped_cells:
            return

        min_distance = float('inf')
        closest_cell = None

        for cell in self.unmapped_cells:
            distance = abs(cell[0] - self.current_position[0]) + abs(cell[1] - self.current_position[1])
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell

        path = self.a_star(self.current_position, closest_cell)
        self.frontier = (closest_cell, path)

        if closest_cell in self.unmapped_cells:
            self.unmapped_cells.remove(closest_cell)

    def a_star(self, start, goal):
        """A* pathfinding algorithm using Manhattan distance heuristic."""
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            for neighbor in self.map.get(current, []):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path
        if goal not in came_from:
            return []

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if self.lidar:
            self.mapping()
        self.find_frontier()

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
