"""EX04: PID Control."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05

        self.left_target_speed = 0.0
        self.right_target_speed = 0.0

        self.prev_left_error = 0.0
        self.prev_right_error = 0.0

        self.integral_left = 0.0
        self.integral_right = 0.0

        self.previous_time = 0.0

    def set_pid(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05) -> None:
        """Set the PID controller gains for the robot's wheel speed control.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_target_speeds(self, left_target: float, right_target: float) -> None:
        """Set the target speeds for the robot's wheels.

        Args:
            left_target (float): Target speed for the left wheel.
            right_target (float): Target speed for the right wheel.
        """
        self.left_target_speed = left_target
        self.right_target_speed = right_target

    def update_left_wheel_speed(self) -> None:
        """Update left wheel speed using PID control."""
        current_speed = self.get_pid_corrected_left_wheel_speed()

        current_time = self.robot.get_time()
        delta_time = current_time - self.previous_time
        if delta_time > 0:
            error = (self.left_target_speed - current_speed) / delta_time
        else:
            error = 0.0

        # P osa pidist proportional gain * error = proportional term
        P_pid = self.kp * error

        # I osa pidist integral term
        self.integral_left += error  # kogunev error, iga kord kui runnib error suureneb
        I_pid = self.ki * self.integral_left  # integral gain korrutada koguneva erroriga = i

        # d osa pidist derivative term
        derivative = error - self.prev_left_error  # kui palju error on eelisest errorist erinev
        if delta_time > 0:
            D_pid = (self.kd * derivative) / delta_time  # derative gain korda errori muutus
        else:
            D_pid = 0.0
        self.prev_left_error = error  # jargmise calli jaoks salvesta error

        correction = P_pid + I_pid + D_pid  # liida koik kokku et saada palju correctima peab
        self.robot.set_left_motor_encoder_ticks(current_speed + correction)  # apply changes

        self.previous_time = current_time

    def update_right_wheel_speed(self) -> None:
        """Update right wheel speed using PID control."""
        current_speed = self.get_pid_corrected_right_wheel_speed()

        current_time = self.robot.get_time()
        delta_time = current_time - self.previous_time
        if delta_time > 0:
            error = (self.right_target_speed - current_speed) / delta_time
        else:
            error = 0.0

        P_pid = self.kp * error

        self.integral_right += error
        I_pid = self.ki * self.integral_right

        derivative = error - self.prev_right_error
        if delta_time > 0:
            D_pid = (self.kd * derivative) / delta_time
        else:
            D_pid = 0.0
        self.prev_right_error = error

        correction = P_pid + I_pid + D_pid
        self.robot.set_right_motor_encoder_ticks(current_speed + correction)

        self.previous_time = current_time

    def get_pid_corrected_left_wheel_speed(self) -> float:
        """Return the corrected left wheel speed."""
        return self.robot.get_left_motor_encoder_ticks()

    def get_pid_corrected_right_wheel_speed(self) -> float:
        """Return the corrected right wheel speed."""
        return self.robot.get_right_motor_encoder_ticks()

    def sense(self) -> None:
        """Gather sensor data."""

    def plan(self) -> None:
        """Plan robot actions."""
        self.update_right_wheel_speed()
        self.update_left_wheel_speed()

    def act(self) -> None:
        """Execute planned actions."""

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()
