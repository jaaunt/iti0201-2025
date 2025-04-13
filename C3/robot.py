import numpy as np


class Robot:
    """Turtlebot robot."""

    STATE_START = "START"
    STATE_SEARCH = "FIND"
    STATE_ALIGN = "ALIGN_TO_OBJ"
    STATE_DRIVE = "APPROACH"
    STATE_STOP = "ARRIVED"
    STATE_FINAL_APPROACH = "FINAL_APPROACH"

    def __init__(self, robot_interface: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot_interface
        self.image = None
        self.detected_target = None
        self.current_time = 0
        self.previous_time = 0
        self.final_approach_start_time = 0

        self.left_motor_speed = -1
        self.right_motor_speed = 1

        self.state = self.STATE_START
        self.object_boxes = []

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.image = self.robot.get_camera_rgb_image()
        self.detected_target = self._get_cube_angle_and_size()
        self.current_time = self.robot.get_time()

    def plan(self):
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        match self.state:
            case self.STATE_START:
                self.left_motor_speed = 1
                self.right_motor_speed = 1
                if self.current_time > 2.5:
                    self.state = self.STATE_SEARCH

            case self.STATE_SEARCH:
                self.left_motor_speed = -1
                self.right_motor_speed = 1
                if self.detected_target:
                    angle, _ = self.detected_target
                    if angle is not None:
                        self.state = self.STATE_ALIGN if abs(angle) > 0.1 else self.STATE_DRIVE

            case self.STATE_ALIGN:
                self.left_motor_speed = -1
                self.right_motor_speed = 1
                if self.detected_target:
                    angle, _ = self.detected_target
                    if abs(angle) < 0.1:
                        self.state = self.STATE_DRIVE

            case self.STATE_DRIVE:
                self._drive_to_target()

            case self.STATE_FINAL_APPROACH:
                self.left_motor_speed = 1
                self.right_motor_speed = 1
                if self.current_time - self.final_approach_start_time > 1.0:
                    self.state = self.STATE_STOP
                    print("Robot on nüüd täielikult peatunud.")

            case self.STATE_STOP:
                self.left_motor_speed = 0
                self.right_motor_speed = 0
                print("Robot on peatunud.")

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        self.robot.set_left_motor_velocity(self.left_motor_speed)
        self.robot.set_right_motor_velocity(self.right_motor_speed)

    def _drive_to_target(self):
        if self.detected_target:
            self.left_motor_speed = 1
            self.right_motor_speed = 1
            self.previous_time = self.current_time
        else:
            self.left_motor_speed = 1
            self.right_motor_speed = 1
            if self.current_time - self.previous_time > 3.0:
                self.state = self.STATE_FINAL_APPROACH
                self.final_approach_start_time = self.current_time
                print("Kuup kadus, liigun veel 1 sekundi edasi.")

    def _get_cube_angle_and_size(self) -> list | None:
        boxes = self._find_object_boxes()
        if boxes is None:
            return None

        try:
            fov = self.robot.get_camera_field_of_view()
        except AttributeError:
            fov = 0.784

        for x_min, x_max, y_min, y_max in boxes:
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            aspect_ratio = width / height

            blue = self.image[y_min:y_max + 1, x_min:x_max + 1, 0]
            green = self.image[y_min:y_max + 1, x_min:x_max + 1, 1]
            red = self.image[y_min:y_max + 1, x_min:x_max + 1, 2]

            mask = (blue > green + 15) & (blue > red + 15)
            if np.sum(mask) / (width * height) > 0.4 and 0.8 <= aspect_ratio <= 1.4:
                center_x = (x_min + x_max) / 2
                _, frame_width, _ = self.image.shape
                size = (x_max - x_min) ** 2
                angle = (center_x - frame_width / 2) * (fov / frame_width)
                return angle, size

    def _find_object_boxes(self) -> list | None:
        if self.image is None:
            return None

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]

        mask = (blue > green + 10) & (blue > red + 10)
        labeled, count = self._label_connected_regions(mask)

        if count == 0:
            return None

        boxes = []
        for i in range(1, count + 1):
            ys, xs = np.where(labeled == i)
            boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))

        return boxes if boxes else None

    def _label_connected_regions(self, binary_mask):
        height, width = binary_mask.shape
        labeled = np.zeros_like(binary_mask, dtype=np.uint32)
        label = 1
        to_process = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y, x in np.argwhere(binary_mask):
            if labeled[y, x] == 0:
                labeled[y, x] = label
                to_process.append((y, x))

                while to_process:
                    cy, cx = to_process.pop()
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary_mask[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = label
                                to_process.append((ny, nx))

                label += 1

        return labeled, label - 1