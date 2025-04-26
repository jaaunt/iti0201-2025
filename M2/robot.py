import math

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

        # Avause ja exit tuvastamise muutujad
        self.left_gap_detected = False
        self.gap_close_counter = 0
        self.first_left_done = False
        self.check_exit_mode = False
        self.exit_check_started = False

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

        print(f"center={self.ir_center:.1f} | left={self.ir_left:.1f} | right={self.ir_right:.1f} | state={self.state} | orientation={math.degrees(self.orientation):.1f}°")

    def handle_state(self):
        if self.check_exit_mode:
            self.handle_exit_check()
            return

        if self.state == "drive":
            if self.ir_center > 50:
                self.state = "turn_right"
                self.turn_start_orientation = self.orientation
                self.orientation_goal = (self.orientation - math.pi/2) % (2*math.pi)

            elif not self.left_gap_detected and self.ir_left > 150:
                self.left_gap_detected = True
                self.gap_close_counter = 0


            elif self.left_gap_detected:
                if self.ir_left < 20:
                    self.gap_close_counter += 1
                if self.gap_close_counter >= 30:
                    if not self.first_left_done:
                        self.state = "turn_left"
                        self.turn_start_orientation = self.orientation
                        self.orientation_goal = (self.orientation + math.pi/2) % (2*math.pi)
                        self.first_left_done = True
                    else:
                        # Teine auk - hakka kontrollima EXITi
                        self.check_exit_mode = True
                    self.left_gap_detected = False
                    self.gap_close_counter = 0

        elif self.state == "turn_left" or self.state == "turn_right":
            if self.reached_orientation():
                self.state = "drive"

    def handle_exit_check(self):
        if not self.exit_check_started:
            # Hakka tegema 180 kraadi pööret
            self.turn_start_orientation = self.orientation
            self.orientation_goal = (self.orientation - math.pi) % (2*math.pi)
            self.state = "exit_turn"
            self.exit_check_started = True

        if self.state == "exit_turn":
            self.setpointL = 1
            self.setpointR = -1
            self.limit = 0.1
            if self.reached_orientation():
                # Kui 180 kraadi tehtud, vaata kas otse on IR väärtused 12
                if self.ir_center < 15 and self.ir_left < 15 and self.ir_right < 15:
                    self.state = "stop"
                    print("Exit tuvastatud! Seisma!")
                else:
                    # Polnud veel väljas, mine tagasi drive!
                    self.check_exit_mode = False
                    self.exit_check_started = False
                    self.state = "drive"

    def reached_orientation(self):
        angle_error = (self.orientation_goal - self.orientation + math.pi) % (2 * math.pi) - math.pi
        return abs(angle_error) < math.radians(5)

    def plan(self) -> None:
        self.handle_state()
        if self.state == "drive":
            self.drive_to_target()
        elif self.state == "turn_left":
            self.turn_left()
        elif self.state == "turn_right":
            self.turn_right()
        elif self.state == "exit_turn":
            # eraldi 180 kraadi keeramiseks
            self.setpointL = 1
            self.setpointR = -1
            self.limit = 0.1
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

    def turn_right(self):
        self.setpointL = 1
        self.setpointR = -1
        self.limit = 0.1

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
