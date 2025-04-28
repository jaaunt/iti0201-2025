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

        self.kp = 0.1
        self.ki = 0.001
        self.kd = 0.001
        self.setpointL = 0
        self.setpointR = 0
        self.limit = 0.05

        self.LeftTicks = [0, 0]
        self.RightTicks = [0, 0]
        self.LeftSpeed = 0
        self.RightSpeed = 0
        self.time_memory = [0, 0]
        self.previous_error_left = 0
        self.previous_error_right = 0
        self.error_sum_left = 0
        self.error_sum_right = 0

        self.orientation = 0

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

        If theres a wall in the way turn right.
        If theres a gap in the left walll turn left.
        Otherwise keep driving straight.
        """
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
        """Drive to the target."""
        self.setpointL = 5
        self.setpointR = 5
        self.limit = 0.05

    def turn_left(self):
        """Turn left."""
        self.setpointL = -1
        self.setpointR = 1
        self.limit = 0.1
        print("turn left")

    def turn_right(self):
        """Turn right."""
        self.setpointL = 1
        self.setpointR = -1
        self.limit = 0.1
        print("turn right")

    def stop(self):
        """Stop the robot."""
        self.setpointL = 0
        self.setpointR = 0
        self.limit = 0.05
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
