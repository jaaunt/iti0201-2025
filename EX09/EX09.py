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

        # cells
        self.traversable_cells = [(0, 0)]
        self.unmapped_cells = []
        self.map = {}

        # data you get from sense
        self.lidar = None
        self.orientation = None

        # other variables
        self.current_position = None  # current (x, y) grid position
        self.frontier = None   # the target unmapped cell and the path to it

        # constants
        self.CELL_SIZE = 0.625
        self.DISTANCE_THRESHOLD = 0.5

    def get_traversable_cells(self) -> list:
        """Return known traversable cells."""
        return self.traversable_cells

    def get_unmapped_cells(self) -> list:
        """Return cells discovered but not yet mapped with LIDAR."""
        return self.unmapped_cells

    def get_map(self) -> dict:
        """Return the map as a dictionary of cell adjacencies."""
        return self.map

    def manhattan(self, a, b):
        """Calculate Manhattan distance between two cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
            self.left = self.lidar[320]  # left (90 degrees)
            self.right = self.lidar[1]  # right (270 degrees)

    def add_cells(self, cell, direction):
        """Add traversable cells in a given direction from current position."""
        x, y = self.current_position  # start from current position
        directions = {  # possible directions and how the values would have to change for them
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        if direction not in directions:  # incase theres an odd case that isnt mentioned in directions
            return

        dx, dy = directions[direction]  # get the direction coordinates based on which way it is

        for step in range(1, cell + 1):
            next_cell_coords = (x + dx * step, y + dy * step)

            # for the first step link it to the robots current position
            if step == 1:
                self.map.setdefault(self.current_position, []).append(next_cell_coords)
                self.map.setdefault(next_cell_coords, []).append(self.current_position)

            # otherwise add it if its not seen yet
            if next_cell_coords not in self.traversable_cells:
                self.traversable_cells.append(next_cell_coords)
                self.unmapped_cells.append(next_cell_coords)

    def mapping(self):
        """Map the environment based on current orientation and LIDAR readings."""
        orientation = self.orientation  # robots current orientation

        lidar_distances = {  # lidar readings for each direction of the robot
            "front": self.front,
            "back": self.back,
            "left": self.left,
            "right": self.right,
        }

        # when the robot is going around where ud down left right is changes
        # consider robots position and where its looking to decide where the maps up would be to the robot
        if -0.1 < orientation < 0.1:  # facing north
            direction_map = {
                "front": "up",
                "back": "down",
                "left": "left",
                "right": "right",
            }
        elif 1.47 < orientation < 1.67:  # facing west
            direction_map = {
                "front": "left",
                "back": "right",
                "left": "down",
                "right": "up",
            }
        elif -1.67 < orientation < -1.47:  # facing east
            direction_map = {
                "front": "right",
                "back": "left",
                "left": "up",
                "right": "down",
            }
        elif abs(abs(orientation) - math.pi) < 0.1:  # facing south
            # anything thats wasnt the earlier, probably south
            # abs so it works for negative values too
            direction_map = {
                "front": "down",
                "back": "up",
                "left": "right",
                "right": "left",
            }
        else:
            return  # catch any orientation related errors, just dont map when it isnt one of the valid ones

        for raw_direction, mapped_direction in direction_map.items():  # raw robots direction mapped what it is for the map
            distance = lidar_distances[raw_direction]  # get the lidar distance in that direction
            if distance > self.DISTANCE_THRESHOLD:  # if its bigger than 0.5 theres probably enough space for it to be a cell
                num_cells = int(distance // self.CELL_SIZE)  # how many cells in that direction
                self.add_cells(num_cells, mapped_direction)

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
        """Select the nearest unmapped cell, frontier, and plan a path to it."""
        if not self.unmapped_cells:  # if it isnt an unmapped cell dont do anything
            return

        min_distance = float('inf')  # since havent found yet, start in a case where the min distance is too far, so it gets overwritten for sure
        closest_cell = None

        # pick the closest unmapped cell using Manhattan distance
        for cell in self.unmapped_cells:
            distance = abs(cell[0] - self.current_position[0]) + abs(cell[1] - self.current_position[1])  # manhatten distance
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell

        # plan a path using a* algorithm
        path = self.a_star(self.current_position, closest_cell)
        self.frontier = (closest_cell, path)

        # remove from list once its being planned to explore
        if closest_cell in self.unmapped_cells:
            self.unmapped_cells.remove(closest_cell)

    def a_star(self, start, goal):
        """Compute the shortest path from start to goal using the A* algorithm."""
        frontier = PriorityQueue()
        frontier.put((0, start))  # start from current position
        came_from = {start: None}  # track how we reached each cell
        cost_so_far = {start: 0}  # track the cost to reach each cell

        while not frontier.empty():  # only works when there are frontiers that havent been explored
            _, current = frontier.get()  # priorityque gets 2 values priority and the item
            # this priority doesnt matter to me since i calculate my own priority, i only need which cell it is

            if current == goal:  # if you reached the goal already stop
                break

            # loop through all neighboring cells of the current cell
            for neighbor in self.map.get(current, []):
                new_cost = cost_so_far[current] + 1  # add the movement cost(its always 1)

                # check if this is the first time we see this neighboring cell,
                # or if its a shorter path to it
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost  # update the cost to reach this neighbour cell
                    # calculate priority like this = total cost so far + manhattan distance)
                    priority = new_cost + self.manhattan(neighbor, goal)
                    # add this neighbor cell to the frontier to be explored ordered by priority
                    frontier.put((priority, neighbor))
                    # remember how we got to this neighbor for path reconstruction later
                    came_from[neighbor] = current

        # reconstruct path
        if goal not in came_from:
            return []

        path = []  # store path here
        current = goal  # start going back from the goal cell
        # keep going backwards through came_from map until we reach the starting cell
        while current is not None:
            path.append(current)
            current = came_from[current]
        # since the path is though backtrackin its in reverse
        # so you have to reverse it to be the right way
        path.reverse()
        return path

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if self.lidar:  # if theres lidar data start mapping
            self.mapping()
        self.find_frontier()  # find the next frontier

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
