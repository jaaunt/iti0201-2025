import math

class Robot:
    def __init__(self, robot):
        self.robot = robot

        # Navigation state
        self.state = "drive"
        self.turn_direction = None
        self.orientation = "N"
        self.position = (0, 0)
        self.visited = {(0, 0)}
        self.path_stack = []
        self.backtrack_target = None

        # Define end goal (UPDATE if needed)
        self.goal_position = (4, -3)  # ← VAJADUSEL MUUDA!

        # IR sensor values
        self.ir = []
        self.ir_center = 0

        # PID motor setup
        self.setpointL = 0
        self.setpointR = 0
        self.kp = 0.1
        self.ki = 0.001
        self.kd = 0.001
        self.limit = 0.05

        self.LeftTicks = [0, 0]
        self.RightTicks = [0, 0]
        self.LeftSpeed = 0
        self.RightSpeed = 0
        self.previous_error_left = 0
        self.previous_error_right = 0
        self.error_sum_left = 0
        self.error_sum_right = 0
        self.time_memory = [0, 0]
        self.dt = 0.01

    def sense(self):
        self.ir = self.robot.get_ir_intensities_list()
        self.ir_center = self.ir[3]
        self.track_speed()

    def track_speed(self):
        self.LeftTicks[0] = self.LeftTicks[1]
        self.RightTicks[0] = self.RightTicks[1]
        self.LeftTicks[1] = self.robot.get_left_motor_encoder_ticks()
        self.RightTicks[1] = self.robot.get_right_motor_encoder_ticks()
        self.time_memory[0] = self.time_memory[1]
        self.time_memory[1] = self.robot.get_time()
        self.dt = self.time_memory[1] - self.time_memory[0]
        if self.dt > 0:
            self.LeftSpeed = (self.LeftTicks[1] - self.LeftTicks[0]) * (2 * math.pi / 508.8) / self.dt
            self.RightSpeed = (self.RightTicks[1] - self.RightTicks[0]) * (2 * math.pi / 508.8) / self.dt
        else:
            self.LeftSpeed = self.RightSpeed = 0

    def plan(self):
        # Stop if at goal and IR says open
        if self.position == self.goal_position and all(ir < 15 for ir in self.ir):
            self.state = "stop"
            return

        if self.state == "drive":
            self.move_forward()
        elif self.state == "turn":
            self.decide_turn()
        elif self.state == "backtrack":
            self.handle_backtrack()

    def move_forward(self):
        x, y = self.position
        if self.orientation == "N":
            new_pos = (x, y + 1)
        elif self.orientation == "E":
            new_pos = (x + 1, y)
        elif self.orientation == "S":
            new_pos = (x, y - 1)
        else:  # "W"
            new_pos = (x - 1, y)

        # If wall ahead or visited, transition to turn or backtrack
        if self.ir_center > 100 or new_pos in self.visited:
            # Try left or right
            if self.ir[0] < 100 and self.get_new_cell("L") not in self.visited:
                self.turn_direction = "left"
                self.update_orientation("L")
                self.state = "turn"
            elif self.ir[6] < 100 and self.get_new_cell("R") not in self.visited:
                self.turn_direction = "right"
                self.update_orientation("R")
                self.state = "turn"
            else:
                self.state = "backtrack"
                if self.path_stack:
                    self.backtrack_target = self.path_stack.pop()
                    self.set_orientation_toward(self.backtrack_target)
                else:
                    self.state = "stop"
            return

        # Move forward
        self.path_stack.append(self.position)
        self.visited.add(new_pos)
        self.position = new_pos

    def decide_turn(self):
        self.state = "drive"

    def handle_backtrack(self):
        if self.backtrack_target:
            self.position = self.backtrack_target
            self.backtrack_target = None
            self.state = "drive"

    def get_new_cell(self, turn):
        x, y = self.position
        ori = self.orientation
        dir_map = {
            ("N", "L"): (x - 1, y),
            ("N", "R"): (x + 1, y),
            ("E", "L"): (x, y + 1),
            ("E", "R"): (x, y - 1),
            ("S", "L"): (x + 1, y),
            ("S", "R"): (x - 1, y),
            ("W", "L"): (x, y - 1),
            ("W", "R"): (x, y + 1),
        }
        return dir_map[(ori, turn)]

    def set_orientation_toward(self, target):
        x, y = self.position
        tx, ty = target
        dx = tx - x
        dy = ty - y
        if dx == 1:
            self.orientation = "E"
        elif dx == -1:
            self.orientation = "W"
        elif dy == 1:
            self.orientation = "N"
        elif dy == -1:
            self.orientation = "S"

    def update_orientation(self, turn):
        dirs = ["N", "E", "S", "W"]
        idx = dirs.index(self.orientation)
        if turn == "L":
            self.orientation = dirs[(idx - 1) % 4]
        else:
            self.orientation = dirs[(idx + 1) % 4]

    def act(self):
        if self.state == "drive":
            self.setpointL = 5
            self.setpointR = 5
        elif self.state == "turn":
            self.setpointL = -1 if self.turn_direction == "left" else 1
            self.setpointR = 1 if self.turn_direction == "left" else -1
        else:  # stop or backtrack
            self.setpointL = 0
            self.setpointR = 0

        torqueL = self.update_pid("L")
        torqueR = self.update_pid("R")
        self.robot.set_left_motor_torque(torqueL)
        self.robot.set_right_motor_torque(torqueR)

    def update_pid(self, side):
        if side == "L":
            setpoint, speed, prev_err, err_sum = self.setpointL, self.LeftSpeed, self.previous_error_left, self.error_sum_left
        else:
            setpoint, speed, prev_err, err_sum = self.setpointR, self.RightSpeed, self.previous_error_right, self.error_sum_right

        error = setpoint - speed
        err_sum += error * self.dt
        d_error = (error - prev_err) / self.dt if self.dt > 0 else 0
        u = self.kp * error + self.ki * err_sum + self.kd * d_error
        u = max(min(u, self.limit), -self.limit)

        if side == "L":
            self.previous_error_left = error
            self.error_sum_left = err_sum
        else:
            self.previous_error_right = error
            self.error_sum_right = err_sum

        return u

    def spin(self):
        self.sense()
        self.plan()
        self.act()
