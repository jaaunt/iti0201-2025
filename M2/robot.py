"""M2."""
import math


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
        self.RADS_PER_TICK = 2 * math.pi / 508.8

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
        self.state = "drive"
        self.turn_direction = "left"
        self.stop_check = False
        self.ticks_check = 0
        self.turn_start_orientation = 0
        self.orientation_goal = 0

        self.ir = []
        self.ir_left = 0.0
        self.ir_center = 0.0
        self.ir_right = 0.0

        self.left_gap_detected = False
        self.gap_close_counter = 0
        self.left_turn_counter = 0

        self.check_camera_after_turn = False
        self.black_before_turn = False

        self.stop_timer_start = None
        self.stop_drive_duration = 1.5

        # pid variables
        self.right_pid = PID()
        self.left_pid = PID()
        self.dt = 0
        self.time_memory = [0, 0]
        self.drive_count = 0

        self.orientation = 0

    def update_limits(self, limit):
        """Update limits of PID controllers."""
        self.left_pid.limit = limit
        self.right_pid.limit = limit

    def snap_to_nearest_90(self, angle_rad):
        """Fix the angle to snapping to the nearest 90 value 0, 90, 180, 270 or 360 degrees."""
        angle_deg = math.degrees(angle_rad)
        snapped_deg = round(angle_deg / 90) * 90
        snapped_deg = snapped_deg % 360
        return math.radians(snapped_deg)

    def get_orientation(self):
        """Get the robots orientation."""
        orientation = self.robot.get_orientation()
        if orientation < 0:
            orientation += 2 * math.pi
        return orientation

    def track_speed(self):
        """Track speed by looking at changes in encoder values."""
        self.left_pid.set_ticks(self.robot.get_left_motor_encoder_ticks())
        self.right_pid.set_ticks(self.robot.get_right_motor_encoder_ticks())

        self.time_memory[0] = self.time_memory[1]
        self.time_memory[1] = self.robot.get_time()
        self.dt = self.time_memory[1] - self.time_memory[0]

        self.left_pid.calculate_speed(self.dt)
        self.right_pid.calculate_speed(self.dt)

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.track_speed()
        self.ir = self.robot.get_ir_intensities_list()
        self.ir_left = self.ir[0]
        self.ir_center = self.ir[3]
        self.ir_right = self.ir[6]
        self.orientation = self.get_orientation()

        print(
            f"center={self.ir_center:.1f} | left={self.ir_left:.1f} | right={self.ir_right:.1f} | state={self.state} | orientation={math.degrees(self.orientation):.1f}°")

    def check_exit_with_lidar(self):
        """Check the lidar distances 180 degrees in front.

        If its mostly inf the robot is almost out and should do a last little sprint to end it a little further from the exit.
        """
        lidar = self.robot.get_lidar_range_list()
        forward_indices = list(range(320, 640))
        forward_distances = [lidar[i] for i in forward_indices]
        inf_count = sum(1 for d in forward_distances if math.isinf(d))
        print(inf_count)
        if inf_count >= (2 / 3) * len(forward_distances):
            self.stop_timer_start = self.robot.get_time()
            self.state = "stop"
        else:
            self.state = "drive"

    def handle_state(self):
        """Handle the robots different states.

        Driving and turning both ways.
        """
        if self.state == "drive":
            self.check_exit_with_lidar()
            self.handle_drive_logic()
        elif self.state in ["turn_left", "turn_right"]:
            self.handle_turn_logic()

    def handle_drive_logic(self):
        """Handle drive logic.

        Fix the damn driving angle.
        If theres a wall in the way turn right.
        If theres a gap in the left wall turn left.
        Otherwise keep driving straight.
        """
        snapped = self.snap_to_nearest_90(self.orientation)
        angle_error = (snapped - self.orientation + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_error) > math.radians(1):  # if the angle is off more than 1 degree
            if angle_error > 0:
                self.state = "turn_left"
            else:
                self.state = "turn_right"
            self.turn_start_orientation = self.orientation
            self.orientation_goal = snapped
            print(f"Correcting driving direction by {math.degrees(angle_error):.1f} degrees.")
            return

        if self.ir_center > 50:
            self.state = "turn_right"
            self.turn_start_orientation = self.orientation
            self.orientation_goal = self.snap_to_nearest_90(self.orientation - math.pi / 2)
        elif not self.left_gap_detected and self.ir_left > 50:
            self.left_gap_detected = True
            self.gap_close_counter = 0
        elif self.left_gap_detected:
            if self.ir_left < 20:
                self.gap_close_counter += 1
            if self.gap_close_counter >= 35:
                if self.left_turn_counter >= 6:
                    if self.ir_center < 50:
                        self.state = "turn_right"
                        self.turn_start_orientation = self.orientation
                        self.orientation_goal = self.snap_to_nearest_90(self.orientation - math.pi / 2)
                        self.left_turn_counter = 0
                    else:
                        self.state = "drive"
                else:
                    self.state = "turn_left"
                    self.turn_start_orientation = self.orientation
                    self.orientation_goal = self.snap_to_nearest_90(self.orientation + math.pi / 2)
                    self.left_turn_counter += 1
                self.left_gap_detected = False
                self.gap_close_counter = 0
        else:
            self.state = "drive"

    def handle_turn_logic(self):
        """Handle turning logic.

        If the robot finished turning start driving again.
        If the robot made a right turn reset the left turn counter.
        """
        if self.reached_orientation():
            if self.state == "turn_right":
                self.left_turn_counter = 0
            self.state = "drive"

    def reached_orientation(self):
        """Check if its reached the orientation goal with a difference of 1 degree."""
        angle_error = (self.orientation_goal - self.orientation + math.pi) % (2 * math.pi) - math.pi
        return abs(angle_error) < math.radians(1)

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if self.state == "stop" and self.stop_timer_start is not None:
            elapsed = self.robot.get_time() - self.stop_timer_start
            if elapsed < self.stop_drive_duration:
                self.drive_to_target()
            else:
                self.stop()
        else:
            self.handle_state()
            if self.state == "drive":
                self.drive_to_target()
            elif self.state == "turn_left":
                self.turn_left()
            elif self.state == "turn_right":
                self.turn_right()
            else:
                self.stop()

    def drive_to_target(self):
        """Drive the robot straight towards the target."""
        self.left_pid.set_setpoint(5)
        self.right_pid.set_setpoint(5)
        self.update_limits(0.05)

    def turn_left(self):
        """Turn left."""
        self.left_pid.set_setpoint(1)
        self.right_pid.set_setpoint(-1)
        self.update_limits(0.1)
        print("turn left")

    def turn_right(self):
        """Turn right."""
        self.left_pid.set_setpoint(-1)
        self.right_pid.set_setpoint(1)
        self.update_limits(0.1)
        print("turn right")

    def stop(self):
        """Stop the robot."""
        self.left_pid.set_setpoint(-10 if self.left_pid.get_speed() > 0 else 0)
        self.right_pid.set_setpoint(-10 if self.right_pid.get_speed() > 0 else 0)
        self.update_limits(0.05)
        if self.left_pid.get_speed() < 0.01 and self.right_pid.get_speed() < 0.01:
            self.drive_count += 1
            self.left_pid.set_setpoint(0)
            self.right_pid.set_setpoint(0)
            self.update_limits(0.05)
        print("stop")

    def update_wheel_speedL(self):
        """Update the left wheel speed of the robot."""
        setpoint = self.setpointL
        speed = self.LeftSpeed
        error = setpoint - speed
        self.error_sum_left += error * self.dt
        error_diff = (error - self.previous_error_left) / self.dt if self.dt > 0 else 0
        u = self.kp * error + self.ki * self.error_sum_left + self.kd * error_diff
        self.previous_error_left = error
        return max(min(u, self.limit), -self.limit)

    def update_wheel_speedR(self):
        """Update the right wheel speed of the robot."""
        setpoint = self.setpointR
        speed = self.RightSpeed
        error = setpoint - speed
        self.error_sum_right += error * self.dt if self.dt > 0 else 0
        error_diff = (error - self.previous_error_right) / self.dt
        u = self.kp * error + self.ki * self.error_sum_right + self.kd * error_diff
        self.previous_error_right = error
        return max(min(u, self.limit), -self.limit)

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
