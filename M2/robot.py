import math
import numpy as np

class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.state = "drive"
        self.turn_direction = "left"
        self.stop_check = False
        self.ticks_check = 0
        self.turn_start_orientation = 0
        self.orientation_goal = 0

        # Sensorite muutujad
        self.ir = []
        self.ir_left = 0.0
        self.ir_center = 0.0
        self.ir_right = 0.0

        # Avause ja loopi tuvastamise muutujad
        self.left_gap_detected = False
        self.gap_close_counter = 0
        self.left_turn_counter = 0

        # Kas peale vasakpööret peab tegema kaamera kontrolli
        self.check_camera_after_turn = False

        # Stop timer
        self.stop_timer_start = None
        self.stop_drive_duration = 2.5  # sek

        # Kiiruse ja PID muutujad
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
        angle_deg = math.degrees(angle_rad)
        snapped_deg = round(angle_deg / 90) * 90
        snapped_deg = snapped_deg % 360
        return math.radians(snapped_deg)

    def get_orientation(self):
        orientation = self.robot.get_orientation()
        if orientation < 0:
            orientation += 2 * math.pi
        return orientation

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
        self.ir_left = self.ir[0]
        self.ir_center = self.ir[3]
        self.ir_right = self.ir[6]
        self.orientation = self.get_orientation()

        print(
            f"center={self.ir_center:.1f} | left={self.ir_left:.1f} | right={self.ir_right:.1f} | state={self.state} | orientation={math.degrees(self.orientation):.1f}°")

    def is_camera_mostly_black(self, threshold=0.62):
        image = self.robot.get_camera_rgb_image()
        rgb_image = image[:, :, :3]
        black_pixels = np.all(rgb_image < 30, axis=2)
        black_ratio = np.sum(black_pixels) / (rgb_image.shape[0] * rgb_image.shape[1])
        print(f"[Camera analysis] Black pixel ratio: {black_ratio:.2f}")
        return black_ratio > threshold

    def handle_state(self):
        if all(ir < 10 for ir in self.ir):
            if not self.stop_check:
                self.stop_check = True
                self.ticks_check = self.RightTicks[1] + 1000
            elif self.RightTicks[1] > self.ticks_check:
                self.state = "stop"
            return

        if self.check_camera_after_turn:
            if self.is_camera_mostly_black():
                self.state = "stop"
                self.stop_timer_start = self.robot.get_time()
            else:
                self.state = "drive"
            self.check_camera_after_turn = False
            return

        if self.state == "drive":
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
                if self.gap_close_counter >= 40:
                    if self.left_turn_counter >= 6:
                        # loop detected -> ignore left turn, drive straight
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

        elif self.state == "turn_left" or self.state == "turn_right":
            if self.reached_orientation():
                if self.state == "turn_left":
                    self.check_camera_after_turn = True
                if self.state == "turn_right":
                    self.left_turn_counter = 0
                self.state = "drive"

    def reached_orientation(self):
        angle_error = (self.orientation_goal - self.orientation + math.pi) % (2 * math.pi) - math.pi
        return abs(angle_error) < math.radians(1)

    def plan(self) -> None:
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
        self.setpointL = 5
        self.setpointR = 5
        self.limit = 0.05

    def turn_left(self):
        self.setpointL = -1
        self.setpointR = 1
        self.limit = 0.1
        print("turn left")

    def turn_right(self):
        self.setpointL = 1
        self.setpointR = -1
        self.limit = 0.1
        print("turn right")

    def stop(self):
        self.setpointL = 0
        self.setpointR = 0
        self.limit = 0.05

    def update_wheel_speedL(self):
        setpoint = self.setpointL
        speed = self.LeftSpeed
        error = setpoint - speed
        self.error_sum_left += error * self.dt
        error_diff = (error - self.previous_error_left) / self.dt if self.dt > 0 else 0
        u = self.kp * error + self.ki * self.error_sum_left + self.kd * error_diff
        self.previous_error_left = error
        return max(min(u, self.limit), -self.limit)

    def update_wheel_speedR(self):
        setpoint = self.setpointR
        speed = self.RightSpeed
        error = setpoint - speed
        self.error_sum_right += error * self.dt if self.dt > 0 else 0
        error_diff = (error - self.previous_error_right) / self.dt
        u = self.kp * error + self.ki * self.error_sum_right + self.kd * error_diff
        self.previous_error_right = error
        return max(min(u, self.limit), -self.limit)

    def act(self) -> None:
        left_torque = self.update_wheel_speedL()
        right_torque = self.update_wheel_speedR()
        self.robot.set_left_motor_torque(left_torque)
        self.robot.set_right_motor_torque(right_torque)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()
