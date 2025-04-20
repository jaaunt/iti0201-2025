"""EX09: Mapping the Environment."""
import math
from collections import deque
from queue import PriorityQueue


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.traversable_cells = [(0, 0)]
        self.unmapped_cells = []
        self.map = {}
        self.lidar = None
        self.orientation = None
        self.current_position = None
        self.frontier = None

    def get_traversable_cells(self) -> list:
        return self.traversable_cells

    def get_unmapped_cells(self) -> list:
        return self.unmapped_cells

    def get_map(self) -> dict:
        return self.map

    def sense(self) -> None:
        self.orientation = self.robot.get_orientation()
        self.lidar = self.robot.get_lidar_range_list()
        self.current_position = self.robot.get_current_position()
        if self.lidar:
            self.front = self.lidar[480]
            self.back = self.lidar[150]
            self.right = self.lidar[1]
            self.left = self.lidar[320]

    def add_cells(self, cell, direction):
        x, y = self.current_position
        directions = {
            "up": (0, 1),
            "down": (0, -1),
            "back": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        if direction not in directions:
            return

        dx, dy = directions[direction]

        for c in range(1, cell + 1):
            coord = (x + dx * c, y + dy * c)

            if c == 1:
                self.map.setdefault(self.current_position, []).append(coord)
                self.map.setdefault(coord, []).append(self.current_position)

            if coord not in self.traversable_cells:
                self.traversable_cells.append(coord)
                self.unmapped_cells.append(coord)

    def case1(self):
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
        if -0.1 < self.orientation < 0.1:
            self.case1()
        elif 1.47 < self.orientation < 1.67:
            self.case2()
        elif -1.67 < self.orientation < -1.47:
            self.case3()
        elif self.orientation > (math.pi - 0.1) or self.orientation < (-math.pi + 0.1):
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
        return self.frontier

    def find_frontier(self):
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
        """A* search using Manhattan distance."""
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
        if self.lidar:
            self.mapping()
        self.find_frontier()

    def act(self) -> None:
        pass

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
