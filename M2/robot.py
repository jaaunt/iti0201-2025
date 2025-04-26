"""M2."""
import math


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.state = "drive"
        self.turn_direction = "left"
        self.stop_check = False
        self.ticks_check = 0
        # speed variables
        self.kp = 0.1
        self.ki = 0.001
        self.kd = 0.001
        self.setpointL = 0
        self.setpointR = 0  # Desired rotational speed in ticks per second
        self.dt = 0.001
        self.LeftTicks = [0, 0]
        self.RightTicks = [0, 0]
        self.RightSpeed = 0
        self.LeftSpeed = 0
        self.time_memory = [0, 0]
        self.previous_error_left = 0
        self.previous_error_right = 0
        self.error_sum_left = 0
        self.error_sum_right = 0
        self.driveCount = 0
        self.limit = 0.05
        # sensor variables
        self.ir = []
        self.ir_center = 0
        self.orientation = 0
        self.orientation_goal = 0

    def handle_state(self):
        left_ir = self.ir[0]
        front_ir = self.ir_center
        right_ir = self.ir[6]

        wall_threshold = 15  # mitte 50, 15 on parem IR-de jaoks

        if all(ir < 10 for ir in self.ir):
            if not self.stop_check:
                self.stop_check = True
                self.ticks_check = self.RightTicks[1] + 1000
            elif self.RightTicks[1] > self.ticks_check:
                self.state = "stop"
            return

        if self.state == "drive":
            if left_ir > 15:
                # Vasakul on ilus sein -> sõida edasi
                self.state = "drive"

            elif front_ir > 50:
                # Otse on sein ees -> keera paremale
                self.turn_direction = "right"
                self.orientation_goal = (self.orientation - math.pi / 2) % (2 * math.pi)
                self.turn_start_time = self.robot.get_time()
                self.state = "turn_right"

            elif left_ir < 10:
                # Vasakul on auk -> keera vasakule
                self.turn_direction = "left"
                self.orientation_goal = (self.orientation + math.pi / 2) % (2 * math.pi)
                self.turn_start_time = self.robot.get_time()
                self.state = "turn_left"

    def get_orientation(self):
        orientation = self.robot.get_orientation()
        if orientation < 0:
            orientation += 2 * math.pi
        return orientation

    def reached_orientation(self):
        margin = 0.25
        angle_diff = abs(self.orientation - self.orientation_goal) % (2 * math.pi)
        return angle_diff < margin or angle_diff > (2 * math.pi - margin)

    def update_wheel_speedL(self):
        setpoint = self.setpointL
        speed = self.LeftSpeed
        error = setpoint - speed
        self.error_sum_left += error * self.dt
        error_diff = (error - self.previous_error_left) / self.dt if self.dt > 0 else 0
        u = self.kp * error + self.ki * self.error_sum_left + self.kd * error_diff
        self.previous_error_left = error
        u = max(min(u, self.limit), -self.limit)
        return u

    def update_wheel_speedR(self):
        setpoint = self.setpointR
        speed = self.RightSpeed
        error = setpoint - speed
        self.error_sum_right += error * self.dt
        error_diff = (error - self.previous_error_right) / self.dt if self.dt > 0 else 0
        u = self.kp * error + self.ki * self.error_sum_right + self.kd * error_diff
        self.previous_error_right = error
        u = max(min(u, self.limit), -self.limit)
        return u

    def drive_to_target(self):
        self.setpointL = 5
        self.setpointR = 5
        self.limit = 0.05

    def turn_left(self):
        self.limit = 0.1
        self.setpointL = -1
        self.setpointR = 1

    def turn_right(self):
        self.limit = 0.1
        self.setpointL = 1
        self.setpointR = -1

    def stop(self):
        self.setpointL = -10 if self.LeftSpeed > 0 else 0
        self.setpointR = -10 if self.RightSpeed > 0 else 0
        self.limit = 0.05
        if self.LeftSpeed < 0.01 and self.RightSpeed < 0.01:
            self.driveCount += 1
            self.setpointL = 0
            self.setpointR = 0
            self.limit = 0.05

    def track_speed(self):
        self.LeftTicks[0] = self.LeftTicks[1]
        self.RightTicks[0] = self.RightTicks[1]

        self.LeftTicks[1] = self.robot.get_left_motor_encoder_ticks()
        self.RightTicks[1] = self.robot.get_right_motor_encoder_ticks()

        self.time_memory[0] = self.time_memory[1]
        self.time_memory[1] = self.robot.get_time()
        self.dt = self.time_memory[1] - self.time_memory[0]

        if self.dt != 0:
            self.LeftSpeed = (self.LeftTicks[1] - self.LeftTicks[0]) * (2 * math.pi / 508.8) / self.dt
            self.RightSpeed = (self.RightTicks[1] - self.RightTicks[0]) * (2 * math.pi / 508.8) / self.dt
        else:
            self.LeftSpeed = 0
            self.RightSpeed = 0
            self.dt = 0.01

    def sense(self) -> None:
        self.track_speed()
        self.ir = self.robot.get_ir_intensities_list()
        self.ir_center = self.ir[3]
        self.orientation = self.get_orientation()
        print(self.state)

    def plan(self) -> None:
        self.handle_state()

        if self.state == "turn_left":
            self.turn_left()
        elif self.state == "turn_right":
            self.turn_right()
        elif self.state == "drive":
            self.drive_to_target()
        else:
            self.stop()

    def act(self) -> None:
        left_torque = self.update_wheel_speedL()
        right_torque = self.update_wheel_speedR()
        self.robot.set_left_motor_torque(left_torque)
        self.robot.set_right_motor_torque(right_torque)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
