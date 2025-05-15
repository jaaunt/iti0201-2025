"""M3."""
import math
from queue import PriorityQueue


class PID:
    """PID controller."""

    def __init__(self):
        """Class initializer."""
        self.kp = 0.1
        self.ki = 0.001
        self.kd = 0.001
        self.limit = 0.05
        self.setpoint = 0
        self.ticks = [0, 0]
        self.speed = 0
        self.prev_error = 0
        self.error_sum = 0
        self.TICKS_PER_ROTATION = 508.8
        self.RADS_PER_TICK = 2 * math.pi / self.TICKS_PER_ROTATION

    def get_ticks(self):
        """Get current ticks."""
        return self.ticks[1]

    def set_ticks(self, ticks):
        """Set current ticks and remember last ticks value."""
        self.ticks[0] = self.ticks[1]
        self.ticks[1] = ticks

    def set_setpoint(self, setpoint):
        """Set setpoint."""
        self.setpoint = setpoint

    def get_speed(self):
        """Get current speed."""
        return self.speed

    def update_wheel_speed(self, dt):
        """Update wheel speed using PID control."""
        error = self.setpoint - self.speed
        error_diff = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        self.error_sum += error * dt
        u = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        u = max(min(u, self.limit), -self.limit)
        return u

    def calculate_speed(self, dt):
        """Calculate rotation speed based on encoder readings."""
        self.speed = (self.ticks[1] - self.ticks[0]) * self.RADS_PER_TICK / dt



class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.state = "mapping"
        self.move = False
        self.movement_state = None
        self.goal_ticks = None
        self.goal_direction = None
        self.current_pos = (0, 0)
        self.orientation = None
        self.direction = "up"
        self.lidar = None
        self.dir_lidar = {}
        self.dir_cells = {"up": (0, 1), "right": (1, 0), "down": (0, -1), "left": (-1, 0)}
        self.robot_directions = {"front": "up", "right": "right", "back": "down", "left": "left"}
        # mapping variables
        self.map = {}
        self.unmapped_cells = [(0, 0)]
        self.target = (0, 0)
        self.stop_zone = None
        self.route = None
        self.visual_map = {}
        # pid variables
        self.right_pid = PID()
        self.left_pid = PID()
        self.dt = 0
        self.time_memory = [0, 0]
        # constants
        self.EDGE_LENGTH = 0.615
        self.CENTERING_DISTANCE = 0.3
        self.ANGLE_MARGIN_OF_ERROR = 0.05
        self.DIST_MARGIN_OF_ERROR = 0.01
        self.METERS_PER_TICK = math.pi * self.robot.WHEEL_DIAMETER / self.right_pid.TICKS_PER_ROTATION

    # sense functions
    def get_direction(self):
        """Determine the robot's direction based on its orientation."""
        if -0.02 < self.orientation < 0.02:
            return "up"
        elif -1.59 < self.orientation < -1.55:
            return "right"
        elif abs(abs(self.orientation) - math.pi) < 0.02:
            return "down"
        elif 1.55 < self.orientation < 1.59:
            return "left"
        else:
            return self.direction

    def track_speed(self):
        """Track speed."""
        self.left_pid.set_ticks(self.robot.get_left_motor_encoder_ticks())
        self.right_pid.set_ticks(self.robot.get_right_motor_encoder_ticks())

        self.time_memory[0] = self.time_memory[1]
        self.time_memory[1] = self.robot.get_time()
        self.dt = self.time_memory[1] - self.time_memory[0]

        self.left_pid.calculate_speed(self.dt)
        self.right_pid.calculate_speed(self.dt)

    # plan functions
    def handle_state(self):
        """Change the robot's state if needed."""
        if self.state == "mapping" and self.unmapped_cells == [] and self.stop_zone is not None:
            self.state = "navigating"
            self.set_target(self.stop_zone)
        elif self.state == "navigating" and self.at_stop_zone():
            self.state = "out"
            self.move = True
            self.movement_state = "stopping"

    def check_movement(self):
        """Stop the robot if it has reached its goal."""
        if self.movement_state == "driving_forward" and self.right_pid.get_ticks() >= self.goal_ticks:
            # change cell
            diff = self.dir_cells[self.direction]
            self.current_pos = self.current_pos[0] + diff[0], self.current_pos[1] + diff[1]
            print("MOVED TO", self.current_pos)
            # stop
            self.movement_state = "centering"
            print("CENTERING")
        elif self.movement_state == "turning" and self.direction == self.goal_direction:
            print("TURNED", self.direction)
            self.movement_state = "stopping"
        elif self.movement_state == "centering":
            self.center_in_cell()
        elif self.movement_state == "stopping" and self.stopped():
            print("STOPPED")
            self.move = False

    def center_in_cell(self):
        """Adjust position to center of the current cell."""
        if self.dir_lidar["back"] == self.dir_lidar["front"] == float('inf'):
            self.movement_state = "stopping"
        else:
            current_back = self.dir_lidar["back"] % self.EDGE_LENGTH
            current_front = self.dir_lidar["front"] % self.EDGE_LENGTH
            if math.isnan(current_back):
                error = self.CENTERING_DISTANCE - current_front
            elif math.isnan(current_front):
                error = self.CENTERING_DISTANCE - current_back
            else:
                error = current_front - current_back

            if abs(error) < self.DIST_MARGIN_OF_ERROR:
                self.movement_state = "stopping"
            else:
                direction = 1 if error > 0 else -1
                self.left_pid.set_setpoint(2 * direction)
                self.right_pid.set_setpoint(2 * direction)
                self.update_limits(0.03)

    def stop(self):
        """Stop the robot."""
        self.left_pid.set_setpoint(-10 if self.left_pid.get_speed() > 0 else 0)
        self.right_pid.set_setpoint(-10 if self.right_pid.get_speed() > 0 else 0)
        self.update_limits(0.05)
        if self.left_pid.get_speed() < 0.01 and self.right_pid.get_speed() < 0.01:
            self.left_pid.set_setpoint(0)
            self.right_pid.set_setpoint(0)
            self.update_limits(0.05)

    def set_target(self, target):
        """Set new target and reset route."""
        self.target = target
        self.route = None

    def at_stop_zone(self):
        """Check if robot has made it out of the maze."""
        return self.current_pos == self.stop_zone

    def stopped(self):
        """Check if robot has stopped moving."""
        return self.left_pid.get_speed() == self.right_pid.get_speed() == 0

    def at_target(self):
        """Check if robot has reached its current target."""
        return self.current_pos == self.target

    def map_cell(self):
        """Map the current cell the robot occupies."""
        neighbours = []
        is_stop_zone = 0
        # find traversable cells in each direction
        for direction, lidar in self.dir_lidar.items():
            if lidar > self.EDGE_LENGTH:
                if lidar == float("inf"):
                    is_stop_zone += 1
                diff_x, diff_y = self.dir_cells[self.robot_directions[direction]]
                neighbour = self.current_pos[0] + diff_x, self.current_pos[1] + diff_y
                neighbours.append(neighbour)

        # if lidar shows inf in 3 or more directions, then stop zone has likely been found
        if is_stop_zone >= 3:
            self.stop_zone = self.current_pos
        else:
            for cell in neighbours:  # add each newly discovered cell to unmapped_cells
                if cell not in self.map.keys():
                    self.unmapped_cells.append(cell)

        print("MAPPED", self.current_pos, ":", neighbours)
        self.map[self.current_pos] = neighbours  # add to map
        # the visual part
        self.visual_map[self.current_pos] = " "

        # hallway
        for neighbor in neighbours:
            if neighbor not in self.visual_map:
                self.visual_map[neighbor] = " "

        # walls
        for dir_name, (dx, dy) in self.dir_cells.items():
            neighbor = (self.current_pos[0] + dx, self.current_pos[1] + dy)
            if neighbor not in neighbours and neighbor not in self.map:
                if neighbor not in self.visual_map:
                    self.visual_map[neighbor] = "#"
        self.set_target(self.unmapped_cells.pop())  # set next cell to map

    def move_to_target(self):
        """Move the robot to the target."""
        if self.route is None or not self.route:
            if self.target in self.map[self.current_pos]:  # if the target is a neighbor
                self.route = [self.target]
            else:
                self.find_route()
        next_cell = self.route[0]  # get next cell to move to in route
        print("ROUTE:", self.route)

        # get which direction
        diff = next_cell[0] - self.current_pos[0], next_cell[1] - self.current_pos[1]
        direction = None
        for potential_direction, cell in self.dir_cells.items():
            if cell == diff:
                direction = potential_direction
                break

        if self.direction == direction:
            self.route.pop(0)
            self.move_forward()
        else:
            self.turn(direction)

    def find_route(self):
        """Find route to the target via A* algorithm."""
        frontier = PriorityQueue()
        frontier.put((0, self.current_pos))  # start from current position
        came_from = {self.current_pos: None}  # track how we reached each cell
        cost_so_far = {self.current_pos: 0}  # track the cost to reach each cell

        while not frontier.empty():
            _, current = frontier.get()

            if current == self.target:  # if reached the goal already stop
                break

            # loop through all neighboring cells of the current cell
            for neighbor in self.map.get(current, []):
                new_cost = cost_so_far[current] + 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.manhattan(neighbor, self.target)
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

        # reconstruct path
        if self.target not in came_from:
            print("NO ROUTE FOUND TO TARGET")
            print("CAME FROM", came_from)
            return []

        path = []
        current = self.target
        # keep going backwards through came_from map until we reach the starting cell
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        self.route = path[1:]

    def manhattan(self, a, b):
        """Calculate Manhattan distance between two cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def move_forward(self):
        """Move the robot forward to the next cell."""
        self.move = True
        self.movement_state = "driving_forward"
        self.goal_ticks = self.right_pid.get_ticks() + self.EDGE_LENGTH / self.METERS_PER_TICK

        self.left_pid.set_setpoint(5)
        self.right_pid.set_setpoint(5)
        self.update_limits(0.05)

    def turn(self, direction):
        """Turn a given direction."""
        self.move = True
        self.movement_state = "turning"
        self.goal_direction = direction

        directions = ["up", "right", "down", "left"]
        diff = directions.index(direction) - directions.index(self.direction)
        if diff == -1 or diff == 3:  # turn left, 3 is for up -> left
            self.left_pid.set_setpoint(-1)
            self.right_pid.set_setpoint(1)
        else:
            self.left_pid.set_setpoint(1)
            self.right_pid.set_setpoint(-1)
        self.update_limits(0.1)

    def update_limits(self, limit):
        """Update limits of PID controllers."""
        self.left_pid.limit = limit
        self.right_pid.limit = limit

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.orientation = self.robot.get_orientation()

        self.direction = self.get_direction()
        if self.direction:
            directions = ["up", "right", "down", "left"]
            robot_directions = ["front", "right", "back", "left"]
            index = directions.index(self.direction)
            for direction in robot_directions:
                self.robot_directions[direction] = directions[index]
                index += 1
                index %= len(directions)

        self.track_speed()

        self.lidar = self.robot.get_lidar_range_list()
        if self.lidar:
            self.dir_lidar["front"] = self.lidar[480]  # front (0 degrees)
            self.dir_lidar["back"] = self.lidar[150]  # back (180 degrees)
            self.dir_lidar["left"] = self.lidar[320]  # left (90 degrees)
            self.dir_lidar["right"] = self.lidar[1]  # right (270 degrees)

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        self.handle_state()

        if self.move:  # if currently moving focus on that
            self.check_movement()
            if self.movement_state == "stopping":
                self.stop()
        elif self.state == "mapping":
            if self.at_target():
                print("AT TARGET")
                self.route = None
                self.map_cell()
            else:
                print("NOT AT TARGET")
                self.move_to_target()
        elif self.state == "navigating" and not self.at_stop_zone():
            self.move_to_target()
        elif self.state == "out" and self.stopped():
            self.print_map()
            self.state = "done"

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        self.robot.set_left_motor_torque(self.left_pid.update_wheel_speed(self.dt))
        self.robot.set_right_motor_torque(self.right_pid.update_wheel_speed(self.dt))

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()

    def print_map(self):
        """Print map based on visual_map for display only."""
        all_cells = set(self.visual_map.keys()) | set(self.map.keys())
        for neighbors in self.map.values():
            all_cells.update(neighbors)

        min_x = min(x for x, y in all_cells)
        max_x = max(x for x, y in all_cells)
        min_y = min(y for x, y in all_cells)
        max_y = max(y for x, y in all_cells)

        print("Map")
        for y in range(max_y, min_y - 1, -1):
            row = ""
            for x in range(min_x, max_x + 1):
                pos = (x, y)
                if pos == self.current_pos:
                    row += "R"
                elif pos == (0, 0):
                    row += "S"
                elif self.stop_zone == pos:
                    row += "E"
                elif pos in self.map:
                    row += " "
                elif pos in self.visual_map:
                    row += self.visual_map[pos]
                else:
                    row += "#"
            print(row)