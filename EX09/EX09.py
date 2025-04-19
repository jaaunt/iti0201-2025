"""EX09: Mapping the Environment."""
import math
from queue import PriorityQueue


def find_wall(span):
    """Find the wall in the list."""
    i = 0
    spans = []
    while i < len(span) - 1:
        # cut the list into subspans where sharp jumps in values occur
        if abs(span[i] - span[i + 1]) > 0.25:
            span_one = span[:i + 1]
            span_two = span[i + 1:]
            spans.append(span_one)
            span = span_two
            i = -1
        i += 1
    spans.append(span)

    most_regular = []
    smallest_avg = None
    # find the wall by looking for the span with the most similar values on average
    for s in spans:
        if len(s) > 9 and len(s) >= 2:  # do not look at shorter lists, 9 was chosen through trial and error
            diffs = []
            for i in range(len(s) - 1):
                diffs.append(abs(s[i] - s[i + 1]))
            avg_diff = sum(diffs) / len(diffs)
            if smallest_avg is None or avg_diff < smallest_avg:
                smallest_avg = avg_diff
                most_regular = s

    return most_regular


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.orientation = None
        self.pos = None
        self.lidar = None

        # enviroment stuff
        self.map = {}
        self.mapped_cells = set()
        self.traversable_cells = {(0, 0)}
        self.unmapped_cells = {(0, 0)}  # to make it easier to get the value
        self.frontier = None
        self.path = []

        # constants
        self.EDGE_LENGTH = 0.615
        self.DIRECTIONS = [0, math.pi / 2, math.pi, 3 * math.pi / 2]  # up, left, down, right
        self.LIDAR_STEP = 2 * math.pi / 640  # 360 degrees divided by lidar range list length
        self.BOUND = 30

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
        # traversable_cells = self.get_traversable_cells()
        # return [cell for cell in traversable_cells if cell not in self.mapped_cells]
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

    def get_orientation(self):
        """Tune the orientation."""
        orientation = self.robot.get_orientation()
        if orientation < 0:
            orientation += 2 * math.pi
        return orientation

    def get_directional_lidar(self):
        """Extract lidar readings for up, left, down, right directions."""
        directional_lidar = []
        start_angle = self.orientation - math.pi / 2  # the angle where lidar range list begins, first readings is 90 degrees clockwise from the front
        if start_angle < 0:
            start_angle += 2 * math.pi
        diff = 2 * math.pi - start_angle
        up_index = -int(diff / self.LIDAR_STEP)

        for i in range(4):
            index = up_index - 160 * i
            if index < -640:
                index += 640
            left_bound = max(index - self.BOUND, -640)
            right_bound = min(index + self.BOUND, 640)
            if right_bound > 0:
                span = self.lidar[left_bound:] + self.lidar[:right_bound]
            elif left_bound < -640:
                span = self.lidar[left_bound + 640:] + self.lidar[:right_bound + 640]
            else:
                span = self.lidar[left_bound:right_bound]
            span = [i for i in span if not math.isinf(i)]
            span = find_wall(span)
            if not span:
                directional_lidar.append(0)
            else:
                directional_lidar.append(max(span))
        return directional_lidar

    def map_cell(self):
        """Map the current cell."""
        directional_lidar = self.get_directional_lidar()  # lidar readings for up, left, down, right directions
        # look at the lidar readings in each direction and determine if cells are reachable
        for i in range(4):
            lidar = directional_lidar[i]
            threshold = 0.9
            x, y = self.pos
            step = 1
            max_dist = lidar
            while threshold <= max_dist:
                if i == 0:
                    y += 1
                elif i == 1:
                    x -= 1
                elif i == 2:
                    y -= 1
                elif i == 3:
                    x += 1
                prev = self.pos
                for step in range(1, int(lidar / self.EDGE_LENGTH) + 1):
                    if i == 0:
                        y = self.pos[1] + step
                        x = self.pos[0]
                    elif i == 1:
                        x = self.pos[0] - step
                        y = self.pos[1]
                    elif i == 2:
                        y = self.pos[1] - step
                        x = self.pos[0]
                    else:
                        x = self.pos[0] + step
                        y = self.pos[1]

                    cell = (x, y)
                    self.traversable_cells.add(cell)
                    self.unmapped_cells.add(cell)
                    self.add_to_map(prev, cell)
                    self.add_to_map(cell, prev)
                    prev = cell

    def add_to_map(self, from_cell, to_cell):
        """Add a new map entry."""
        if from_cell not in self.map.keys():
            self.map[from_cell] = [to_cell]
        elif to_cell not in self.map[from_cell]:
            self.map[from_cell].append(to_cell)

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
        if self.frontier is not None and self.path:
            return [self.frontier, self.path]
        return []

    def find_frontiers(self):
        """Find traversable cells next to unknown ones (frontier cells)."""
        frontiers = set()
        for cell in self.mapped_cells:
            x, y = cell
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in self.traversable_cells and neighbor not in self.mapped_cells:
                    frontiers.add(neighbor)
        return list(frontiers)

    def choose_closest_frontier(self, frontiers: list):
        """Find the closest frontier."""
        return min(frontiers, key=lambda cell: (abs(cell[0] - self.pos[0]) + abs(cell[1] - self.pos[1]), cell[1], cell[0]))

    def find_path(self, start: tuple, goal: tuple) -> list:
        """Use A* to find the shortest path from start to goal."""
        def distance_to_goal(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # how long is the distance to the goal, manthattan distance

        frontier = PriorityQueue()  # always get the lowest priority(lowest value and the distance) one first
        frontier.put((0, start))  # start from 0
        came_from = {start: None}  # remember how you got there
        cost_so_far = {start: 0}  # movement cost

        while not frontier.empty():  # while there are frontier cells to check get one with the lowest
            _, current = frontier.get()
            if current == goal:  # if u get to the goal just stop
                break
            for neighbor in self.map.get(current, []):  # look at the neighbours
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:  # update the neighbour if not visited ot found a lower cost path
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + distance_to_goal(goal, neighbor)
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return []  # no path found

        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    def update_frontier_and_path(self):
        """Update the current frontier and the shortest planned path to it."""
        # a lot of print for testing locally
        frontiers = self.find_frontiers()
        print(f"Found frontiers: {frontiers}")

        if not frontiers:
            print("No frontiers left to explore.")
            self.frontier = None
            self.path = []
            return

        self.frontier = self.choose_closest_frontier(frontiers)
        print(f"Chosen frontier: {self.frontier}")
        print(f"Current position: {self.pos}")
        print(f"Map keys: {list(self.map.keys())}")

        path = self.find_path(self.pos, self.frontier)
        print(f"Computed path: {path}")

        if path and len(path) > 1:
            self.path = path
        else:
            print(f"[DEBUG] Path invalid or too short: {path}")
            self.path = []

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.orientation = self.get_orientation()
        self.pos = self.robot.get_current_position()
        self.lidar = self.robot.get_lidar_range_list()

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if self.lidar is not None and self.pos not in self.mapped_cells:
            self.map_cell()

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
        self.update_frontier_and_path()
