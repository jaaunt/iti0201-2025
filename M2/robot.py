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
        self.LeftTicks = [0, 0]  # index 0 on eelmise tsükli omad, index 1 on selle tsükli omad
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
        """Change the robot's state if needed."""
        # if robot is driving forward and approaching a wall, then it's time to turn
        if self.state == "drive" and self.ir_center > 100:
            self.state = "turn"
            if self.ir[0] <= self.ir[6]:  # if left infrared sensor shows smaller value than right sensor turn left
                self.turn_direction = "left"
                self.orientation_goal = self.orientation + math.pi / 2

                if self.orientation_goal > 2 * math.pi:
                    self.orientation_goal -= 2 * math.pi
            else:
                self.turn_direction = "right"
                self.orientation_goal = self.orientation - math.pi / 2

                if self.orientation_goal < 0:
                    self.orientation_goal += 2 * math.pi

        # if the robot has turned 90 degrees to one side, it can start driving forward again
        if self.state == "turn" and self.reached_orientation():
            self.state = "drive"
            self.stop_check = False

        # if all sensors show far, then robot has exited the maze and can stop
        if all(ir < 15 for ir in self.ir):
            if not self.stop_check:
                self.stop_check = True
                self.ticks_check = self.RightTicks[1] + 1000
            elif self.RightTicks[1] > self.ticks_check:
                self.state = "stop"

    def get_orientation(self):
        """Tune the orientation."""
        orientation = self.robot.get_orientation()
        if orientation < 0:
            orientation += 2 * math.pi
        return orientation

    def reached_orientation(self):
        """Check if robot has reached its orientation goal."""
        if self.turn_direction == "left" and self.orientation > self.orientation_goal and self.orientation - self.orientation_goal <= 0.01:
            return True
        elif self.turn_direction == "right" and self.orientation < self.orientation_goal and self.orientation_goal - self.orientation <= 0.01:
            return True
        else:
            return False

    def update_wheel_speedL(self):
        """Update wheel speed using PID control."""
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
        """Update wheel speed using PID control."""
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
        """Drive the robot straight towards the target."""
        self.setpointL = 5
        self.setpointR = 5
        self.limit = 0.05

    def turn(self):
        """Turn the robot to the side."""
        self.limit = 0.1
        if self.turn_direction == "left":
            self.setpointL = -1
            self.setpointR = 1
        else:
            self.setpointL = 1
            self.setpointR = -1

    def stop(self):
        """Stop the robot."""
        self.setpointL = -10 if self.LeftSpeed > 0 else 0
        self.setpointR = -10 if self.RightSpeed > 0 else 0
        self.limit = 0.05
        if self.LeftSpeed < 0.01 and self.RightSpeed < 0.01:
            self.driveCount += 1
            self.setpointL = 0
            self.setpointR = 0
            self.limit = 0.05

    def track_speed(self):
        """Track speed."""
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
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.track_speed()

        self.ir = self.robot.get_ir_intensities_list()
        self.ir_center = self.ir[3]
        self.orientation = self.get_orientation()

        print("LIST:", self.ir)
        print("TICKS:", self.RightTicks)
        # print("CENTER:", self.ir_center)
        print("ORIENTATION:", self.orientation)
        print("STATE:", self.state)

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        self.handle_state()

        if self.state == "turn":
            self.turn()
        elif self.state == "drive":
            self.drive_to_target()
        else:
            self.stop()

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        left_torque = self.update_wheel_speedL()
        right_torque = self.update_wheel_speedR()
        self.robot.set_left_motor_torque(left_torque)
        self.robot.set_right_motor_torque(right_torque)

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
