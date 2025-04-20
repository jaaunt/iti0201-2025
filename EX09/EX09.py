"""EX09: Mapping the Environment."""
import math
from collections import deque


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Initialize the robot."""
        self.robot = robot
        self.map = {}
        self.traversable_cells = {(0, 0)}
        self.unmapped_cells = {(0, 0)}
        self.lidar = None
        self.orientation = None
        self.pos = None
        self.frontier = None

    def get_traversable_cells(self) -> list:
        """Return known traversable cells."""
        return list(self.traversable_cells)

    def get_unmapped_cells(self) -> list:
        """Return discovered but unmapped cells."""
        return list(self.unmapped_cells)

    def get_map(self) -> dict:
        """Return the adjacency map."""
        return self.map

    def sense(self) -> None:
        """Gather sensor data."""
        self.orientation = self.robot.get_orientation()
        self.lidar = self.robot.get_lidar_range_list()
        self.pos = self.robot.get_current_position()
        if self.lidar:
            self.front = self.lidar[480]
            self.back = self.lidar[150]
            self.right = self.lidar[1]
            self.left = self.lidar[320]

    def add_cells(self, count: int, direction: str) -> None:
        """Add reachable cells in a given direction."""
        x, y = self.pos
        directions = {
            "up": (0, 1),
            "down": (0, -1),
            "back": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        dx, dy = directions[direction]

        for i in range(1, count + 1):
            cell = (x + dx * i, y + dy * i)

            if i == 1:
                self.map.setdefault(self.pos, []).append(cell)
                self.map.setdefault(cell, []).append(self.pos)

            if cell not in self.traversable_cells:
                self.traversable_cells.add(cell)
                self.unmapped_cells.add(cell)

    def map_by_direction(self, dir_map):
        """Map reachable cells using direction mapping."""
        readings = {
            "front": self.front,
            "back": self.back,
            "right": self.right,
            "left": self.left
        }
        for sensor, distance in readings.items():
            if distance > 0.45:
                self.add_cells(int(distance // 0.625), dir_map[sensor])

    def mapping(self):
        """Perform mapping based on orientation using direction mapping."""
        if -0.1 < self.orientation < 0.1:
            dir_map = {"front": "up", "back": "down", "right": "right", "left": "left"}
        elif 1.47 < self.orientation < 1.67:
            dir_map = {"front": "left", "back": "right", "right": "up", "left": "down"}
        elif -1.67 < self.orientation < -1.47:
            dir_map = {"front": "right", "back": "left", "right": "down", "left": "up"}
        elif self.orientation > (math.pi - 0.1) or self.orientation < (-math.pi + 0.1):
            dir_map = {"front": "down", "back": "up", "right": "left", "left": "right"}
        else:
            return  # unknown orientation, do nothing

        self.map_by_direction(dir_map)

    def get_frontier_and_path(self) -> list:
        """Return the current frontier and path."""
        return self.frontier

    def find_frontier(self) -> None:
        """Find the closest unmapped cell and path to it."""
        if not self.unmapped_cells:
            return

        min_dist = float("inf")
        closest = None
        for cell in self.unmapped_cells:
            dist = abs(cell[0] - self.pos[0]) + abs(cell[1] - self.pos[1])
            if dist < min_dist:
                min_dist = dist
                closest = cell

        path = self.bfs(self.pos, closest)
        self.frontier = (closest, path)
        self.unmapped_cells.discard(closest)

    def bfs(self, start, goal):
        """Breadth-first search to find a path."""
        queue = deque()
        queue.append((start, [start]))
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            for neighbor in self.map.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def plan(self) -> None:
        """Plan next steps based on mapping and pathfinding."""
        if self.lidar:
            self.mapping()
        self.find_frontier()

    def act(self) -> None:
        """Placeholder for acting."""
        pass

    def spin(self) -> None:
        """Sense, plan, and act loop."""
        self.sense()
        self.plan()
        self.act()
