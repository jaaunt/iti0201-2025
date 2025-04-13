"""C3."""
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.image: np.ndarray | None = None
        self.fov: float | None = None
        self.lidar: list[float] = []
        self.non_blue_detected: bool = False
        self.blue_cubes = None
        self.left_velocity = 0
        self.right_velocity = 0
        self.state = "approaching"
        self.avoid_timer = 0
        self.avoid_cooldown = 0

    def find_blobs(self, mask):
        """Flood fill algorithm to find the blue object."""
        height, width = mask.shape
        labled_mask = np.zeros_like(mask, dtype=np.uint32)
        lable_id = 1
        to_visit = []
        neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for y, x in np.argwhere(mask):
            if labled_mask[y, x] == 0:
                labled_mask[y, x] = lable_id
                to_visit.append((y, x))
                while to_visit:
                    current_y, current_x = to_visit.pop()
                    for dy, dx in neighbours:
                        new_y, new_x = current_y + dy, current_x + dx
                        if 0 <= new_y < height and 0 <= new_x < width:
                            if mask[new_y, new_x] and labled_mask[new_y, new_x] == 0:
                                labled_mask[new_y, new_x] = lable_id
                                to_visit.append((new_y, new_x))
                lable_id += 1

        return labled_mask, lable_id - 1

    def get_object_bounding_box_list(self) -> list | None:
        """Get the object bounding box list."""
        if self.image is None:
            return None

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50

        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        labled_mask, lable_count = self.find_blobs(mask)

        if lable_count == 0:
            return None

        blobs = []
        for i in range(1, lable_count + 1):
            blobs_pixels = np.column_stack(np.where(labled_mask == i))
            if blobs_pixels.size == 0:
                return None

            y_min, x_min = blobs_pixels.min(axis=0)
            y_max, x_max = blobs_pixels.max(axis=0)
            blobs.append((x_min, x_max, y_min, y_max))

        return blobs if blobs else None

    def update_cube_objects(self):
        """Update cubes and return them."""
        boxes = self.get_object_bounding_box_list()
        if boxes is None:
            return None
        cubes = []
        for box in boxes:
            x_min, x_max = box[0], box[1]
            y_min, y_max = box[2], box[3]
            x_side = x_max - x_min
            y_side = y_max - y_min

            threshold = 20
            difference = abs(x_side - y_side)
            if difference <= threshold:
                cubes.append(box)
        return cubes

    def get_cube_objects(self) -> list | None:
        """Get the cubes and return them."""
        self.blue_cubes = self.update_cube_objects()
        return self.blue_cubes if self.blue_cubes else None

    def get_cube_angle(self):
        """Return the horizontal angle (in radians) to the detected blue cube center."""
        if self.image is None or self.fov is None:
            return None

        cube_boxes = self.get_cube_objects()
        if not cube_boxes:
            return None

        height, width = self.image.shape[:2]
        x_min, x_max, _, _ = cube_boxes[0]
        x_center = (x_min + x_max) / 2
        angle = ((x_center - width / 2) / (width / 2)) * (self.fov / 2)
        return angle

    def _handle_approaching(self):
        """Rotate robot to face the blue cube based on camera angle."""
        print("APPROACHING: trying to face the cube")
        angle = self.get_cube_angle()

        if angle is None:
            print("No cube detected – spinning left to search")
            self.left_velocity = -2.5
            self.right_velocity = 2.5
            return

        if abs(angle) < 0.2:
            print("Cube is centered – switching to DRIVING state")
            self.left_velocity = 0
            self.right_velocity = 0
            self.state = "driving"
        elif angle > 0:
            print("Cube is to the right – turning right")
            self.left_velocity = 1.5
            self.right_velocity = -1.5
        else:
            print("Cube is to the left – turning left")
            self.left_velocity = -1.5
            self.right_velocity = 1.5

    def _handle_driving(self):
        cube_boxes = self.get_cube_objects()
        if not cube_boxes:
            print("Lost sight of cube – returning to APPROACHING state")
            self.state = "approaching"
            return

        if self.lidar:
            center_index = len(self.lidar) // 2
            front_distance = self.lidar[center_index]
            if front_distance is not None and front_distance < 0.4:
                print("Obstacle ahead – stopping!")
                self.left_velocity = 0
                self.right_velocity = 0
                return

        x_min, x_max, y_min, y_max = cube_boxes[0]
        box_height = y_max - y_min

        if box_height > 100:
            print("Arrived at the cube!")
            self.left_velocity = 0
            self.right_velocity = 0
        else:
            print("Driving forward toward cube")
            self.left_velocity = 5.0
            self.right_velocity = 5.0

    def _detect_non_blue_object(self):
        if self.image is None:
            return False

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        threshold = 50

        red_object = (red > blue + threshold) & (red > green + threshold)
        yellow_object = (red > blue + threshold) & (green > blue + threshold)

        return np.any(red_object) or np.any(yellow_object)

    def _handle_avoiding(self):
        print("Avoiding non-blue object – turning")

        if self.lidar:
            left = np.mean(self.lidar[:len(self.lidar) // 3])
            right = np.mean(self.lidar[2 * len(self.lidar) // 3:])

            if left > right:
                # turn left if left side is clearer
                self.left_velocity = -2.0
                self.right_velocity = 2.0
            else:
                # turn right otherwise
                self.left_velocity = 2.0
                self.right_velocity = -2.0
        else:
            # default to turning
            self.left_velocity = -2.0
            self.right_velocity = 2.0

        self.avoid_timer -= 1

        if self.avoid_timer <= 0:
            print("Avoid complete – returning to APPROACHING")
            self.state = "approaching"
            self.avoid_cooldown = 40

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.update_cube_objects()

        self.non_blue_detected = self._detect_non_blue_object()

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if not hasattr(self, "state"):
            self.state = "approaching"

        if self.avoid_cooldown > 0:
            self.avoid_cooldown -= 1

        if self.state == "approaching":
            if self.non_blue_detected and self.avoid_cooldown == 0:
                print("Non-blue object detected – switching to AVOIDING")
                self.state = "avoiding"
                self.avoid_timer = 20
            else:
                self._handle_approaching()

        elif self.state == "driving":
            self._handle_driving()

        elif self.state == "avoiding":
            self._handle_avoiding()

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
