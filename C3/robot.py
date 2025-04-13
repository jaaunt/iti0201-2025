"""C3."""
import numpy as np
import math


def simplify_color_data(array, function):
    """Simplify to 0s and 1s."""
    simplified = []
    for y, y_row in enumerate(array):
        row = []
        for pixel in y_row:
            if function(pixel):
                row.append(1)
            else:
                row.append(0)
        simplified.append(row)

    return simplified


def simplify_color_data_optimised(array, color):
    """Simplify to 0s and 1s: 1 if blue, else 0."""
    # Split BGR channels
    blue = array[:, :, 0].astype(float)
    green = array[:, :, 1].astype(float)
    red = array[:, :, 2].astype(float)

    # Apply the logic to entire arrays
    if color == "BLUE":
        mask = (blue > 100) & ((green + red) < 143)
        return mask.astype(int)
    if color == "RED":
        mask = (red > 95) & (60 > green) & (60 > blue)
        return mask.astype(int)
    if color == "YELLOW":
        mask = (red > 110) & (green > 107) & (55 > blue)
        return mask.astype(int)


def is_pixel_blue(bgr_tuple):
    """Return whether we are dealing with a blue pixel."""
    blue = float(bgr_tuple[0])
    green = float(bgr_tuple[1])
    red = float(bgr_tuple[2])

    if blue > 100 and 143 > green + red:  # Good for starters
        # print(f"r: {red}, g: {green}, b: {blue}")
        return True
    return False


def clean_indexes(input_list):
    """
    End with period.

    Given an input consisting of number clusters representing pixel columns of blue objects,
    return the start and finish of each object.
    Example input: [31, 32, 33, 34, 35, 77, 78, 79, 80, 81, 82, 677, 678, 679, 680]
    Example output: [32, 35, 78, 81, 678, 679]
    1 is added to the left side and 1 is substracted from the right to avoid artefacts made by the camera.
    """
    if input_list:
        result = []
        last_element = None
        for i in input_list:
            if not result:
                result.append(i + 1)
            else:
                if i != last_element + 1:
                    result.append(last_element - 1)
                    result.append(i + 1)
            last_element = i

        result.append(last_element - 1)
        return result
    return None


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.orientation = None
        self.angle = None
        self.distance = None
        self.closest_object = None
        self.theta = None
        self.right_motor_speed = None
        self.left_motor_speed = None
        self.object_close = False
        self.current_data = None
        self.data = None

        self.x = 0.0
        self.y = 0.0
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0
        self.right_motor_velocity = 0.0
        self.left_motor_velocity = 0.0
        self.wheel_radius = self.robot.WHEEL_DIAMETER / 2
        self.wheel_base = 0.16
        self.ticks_per_revolution = 508.8
        self.linear_velocity = 0.0

        self.prev_time = self.robot.get_time()
        self.bounding_boxes = {}
        self.fov = None
        self.color_array = []
        self.TESTING = False
        self.spin_count = 1
        self.historic_y_coords = []
        self.historic_angles = []
        self.final_direction = None

    def get_object_location_list(self) -> list | None:
        """Calculate the coordinates for detected object center and corresponding angle.

        Logic:
            - Use camera data to detect blue objects and their bounding boxes. To achieve
                this, analyze the data structure/data channels (hint: look at the names of
                the channels) and find the relevant pixels.
            - For each detected object, it computes the centroid (center) and calculates
                the angle with respect to the robot's camera's field of view (FOV).
            - Finally the angle of the object in terms of the robot's orientation needs
                to be found.

        Returns:
            A list of detected objects, where each object is represented as a list:
            [[x-coordinate of centroid, y-coordinate of centroid, angle with
            respect to robot orientation], ...].
            If no objects are detected, returns None.
        """
        objects = self.get_cube_bounding_box_list()
        detected_objects = []
        if objects is None:
            return None
        camera_width = self.color_array.shape[1]
        camera_center_x = camera_width / 2
        for obj in objects:
            x_min, x_max, y_min, y_max = obj
            x_center_coordinate = (x_min + x_max) / 2
            y_center_coordinate = (y_min + y_max) / 2
            angle = (x_center_coordinate - camera_center_x) / camera_center_x * (self.fov / 2)
            detected_objects.append([x_center_coordinate, y_center_coordinate, angle])
        self.angle = angle
        return detected_objects

    def update_odometry(self):
        """Update the robot's estimated position and orientation (odometry)."""
        self.theta = self.robot.get_orientation()

        current_left_ticks = self.robot.get_left_motor_encoder_ticks()
        current_right_ticks = self.robot.get_right_motor_encoder_ticks()

        left_ticks_delta = current_left_ticks - self.prev_left_ticks
        right_ticks_delta = current_right_ticks - self.prev_right_ticks

        left_distance = (left_ticks_delta / self.ticks_per_revolution) * (2 * math.pi * self.wheel_radius)
        right_distance = (right_ticks_delta / self.ticks_per_revolution) * (2 * math.pi * self.wheel_radius)

        center_distance = (left_distance + right_distance) / 2

        self.x += center_distance * math.cos(self.theta)
        self.y += center_distance * math.sin(self.theta)

        self.prev_left_ticks = current_left_ticks
        self.prev_right_ticks = current_right_ticks

        current_time = self.robot.get_time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        self.left_motor_velocity = left_distance / dt if dt > 0 else 0.0
        self.right_motor_velocity = right_distance / dt if dt > 0 else 0.0
        self.linear_velocity = (self.right_motor_velocity + self.left_motor_velocity) / 2

    def get_cube_bounding_box_list(self) -> list | None:
        """Calculate the bounding box for any detected blue object.

        Logic:
          - This method analyzes the camera image to detect blue-colored objects by
            isolating the blue channel and applying a color threshold to identify blue
            regions.
          - The bounding boxes are returned as a list of tuples, where each tuple is
            (x_min, x_max, y_min, y_max), representing the horizontal and vertical
            limits of the detected object.

        Returns:
          A list of bounding boxes, each bounding box is represented as a tuple
          [(x_min, x_max, y_min, y_max), ...].
          Returns None if no objects are detected.
        """
        if self.color_array is None:
            return None

        # Preprocess that mf
        simplified_version = np.asarray(simplify_color_data_optimised(self.color_array, "BLUE"))
        self.s = simplified_version
        # Get blue objects´ x indexes
        blue_indexes = set()
        for col in range(len(simplified_version[0])):
            if np.any(simplified_version[:, col] == 1) != 0:
                blue_indexes.add(col)
        blue_indexes = clean_indexes(list(sorted(blue_indexes)))  # Every object's side indexes starting from the left

        if not blue_indexes:
            return None

        # For each left side, let's find the min y
        # For each right side, let's find the max y
        bounding_boxes = {}
        column_height = len(simplified_version)
        obj_index = 0
        for n, side_index in enumerate(blue_indexes):
            if n % 2 == 0:  # A new object's left side
                right_side = blue_indexes[n + 1]
                # Let's find the smallest index of a blue pixel between the object's sides
                y_min = min(np.argmax(simplified_version[:, ind] == 1) for ind in range(side_index, right_side))
                obj_index = int(n / 2)
                bounding_boxes[f"obj_{obj_index}_x_min"] = side_index
                bounding_boxes[f"obj_{obj_index}_y_min"] = y_min

                # Now the same for the y_max, this time iterating from below
                index = min(np.argmax(simplified_version[:, ind][::-1] == 1) for ind in range(side_index, right_side))
                y_max = column_height - index - 1
                bounding_boxes[f"obj_{obj_index}_x_max"] = right_side
                bounding_boxes[f"obj_{obj_index}_y_max"] = y_max

        # Return in an appropriate format
        result = []
        for o in range(obj_index + 1):
            x_min = bounding_boxes[f"obj_{o}_x_min"]
            x_max = bounding_boxes[f"obj_{o}_x_max"]
            y_min = bounding_boxes[f"obj_{o}_y_min"]
            y_max = bounding_boxes[f"obj_{o}_y_max"]

            if 0.5 * (x_max - x_min) <= y_max - y_min <= (x_max - x_min) * 1.2:
                result.append((x_min, x_max, y_min, y_max))
                print("                                on kuup")
            else:
                print(f"                                ei ole kuup {(x_min, x_max, y_min, y_max)}")

        return result if result else None  # Mic drop

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.orientation = self.robot.get_orientation()
        self.color_array = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()

        self.update_odometry()

        # if not self.data or len(self.data) == 0:
        #    self.closest_object = None
        #    self.distance = None
        #    self.angle = None
        #    return None

        if self.spin_count % 1 == 0:
            objects = self.get_object_location_list()

            if objects:
                print("objects exist")
                self.closest_object = objects[0]
                x_center, y_center, self.angle = self.closest_object
                # self.distance = 2  # dummy
                if 630 < x_center < 650:
                    self.distance = (900 - y_center) * 0.0061
                    self.historic_y_coords.append(y_center)
                    self.historic_angles.append(self.angle)
            elif 3 < len(self.historic_y_coords):
                if not self.final_direction:
                    self.final_direction = self.robot.get_orientation()
                last = self.historic_y_coords[-1]
                second_to_last = self.historic_y_coords[-2]
                self.historic_y_coords.append(2 * last - second_to_last)  # linear continuation aka for example 4 is added to 1, 2, 3

                self.distance = (900 - self.historic_y_coords[-1]) * 0.0061
                self.angle = -self.final_direction + self.robot.get_orientation()
                self.closest_object = 1, 1, 1
                print(f"continued y is: {self.historic_y_coords[-1]}, fd: {self.final_direction}, orien: {self.robot.get_orientation()}")
            else:
                print(f"Looking around {len(self.historic_y_coords)}")

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        if self.spin_count < 112:
            print(f"spins: {self.spin_count}")
            self.angle = self.robot.get_orientation() - 1.81
            self.distance = 2
            self.closest_object = 1, 1, 1

        print(f"{self.distance}, {self.angle}")
        if self.linear_velocity <= 0:
            self.right_motor_torque = 0.0
            self.left_motor_torque = 0.0
            return
        elif self.angle is None or self.distance is None or self.closest_object is None:
            print(f"shit hit the fan {self.angle}, {self.distance}, {self.closest_object}")
            # elif self.angle is None or self.closest_object is None:
            self.right_motor_torque = -0.05
            self.left_motor_torque = 0.05

        elif self.distance < 0.4:
            self.right_motor_torque = -0.03
            self.left_motor_torque = -0.03
        elif -0.05 < self.angle - self.theta < 0.05:
            if self.distance < 0.4:
                self.right_motor_torque = -0.055
                self.left_motor_torque = -0.055
                return
            else:
                self.right_motor_torque = 0.00105
                self.left_motor_torque = 0.00105

        else:
            if self.angle > 0.0:
                self.right_motor_torque = -0.06
                self.left_motor_torque = 0.06
            else:
                self.right_motor_torque = 0.06
                self.left_motor_torque = -0.06

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        # print("There is movement", self.linear_velocity, self.right_motor_torque, self.left_motor_torque)
        self.robot.set_right_motor_torque(self.right_motor_torque)
        self.robot.set_left_motor_torque(self.left_motor_torque)

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.spin_count += 1
        self.sense()
        self.plan()
        self.act()