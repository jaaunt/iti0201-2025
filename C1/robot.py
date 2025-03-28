"""C1."""
from __future__ import annotations
import numpy as np
import math

class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.detected_objects = []
        self.robot = robot
        self.state = "init"
        self.left_velocity = 0
        self.right_velocity = 0

        self.image = None
        self.fov = None
        self.object = []


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
          If no objects are detected, returns `None`.
        """
        bounding_boxes = self.get_object_bounding_box_list()
        if not bounding_boxes:
            return None

        height, width = self.image.shape
        fov = self.fov

        object_locations = []
        for x_min, x_max, y_min, y_max in bounding_boxes:
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2  # centroid otsimine basiacally see kesk punk obj

            angle = ((x_center - width / 2) / (width / 2)) * (fov / 2)

            object_locations.append([x_center, y_center, angle])
        return object_locations if object_locations else None

    def find_blobs(self, mask):
        """
        Flood fill algorithm to find the blue object.

        :param mask:
        :return:
        """
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
                    for direction_y, direction_x in neighbours:
                        new_y, new_x = current_y + direction_y, current_x + direction_x
                        if 0 <= new_y < height and 0 <= new_x < width:
                            if mask[new_y, new_x] and labled_mask[new_y, new_x] == 0:
                                labled_mask[new_y, new_x] = lable_id
                                to_visit.append((new_y, new_x))
                lable_id += 1

        return labled_mask, lable_id - 1

    def get_object_bounding_box_list(self) -> list | None:
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
          Returns `None` if no objects are detected.
        """
        if self.image is None:
            return None

        blue_channel =self.image[:, :, 0]
        green_channel =self.image[:, :, 1]
        red_channel =self.image[:, :, 2]
        threshold = 50
        mask = (blue_channel > green_channel) + threshold & (blue_channel > red_channel) + threshold

        labled_mask, lable_count = self.find_blobs(mask)
        if lable_count == 0:
            return None
        blobs = []
        for i in range(1, lable_count + 1):
            blobs_pixels = np.column_stack(np.where(labled_mask == i))
            if blobs_pixels.size == 0:
                return None
            print(blobs_pixels.shape)
            print(np.min(blobs_pixels[:, 1]))
            print(np.max(blobs_pixels[:, 1]))
            print(np.min(blobs_pixels[:, 0]))
            print(np.max(blobs_pixels[:, 0]))
            y_min, x_min = blobs_pixels.min(axis=0)
            y_max, x_max = blobs_pixels.max(axis=0)
            blobs.append((x_min, x_max, y_min, y_max))

        return blobs if blobs else None

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()

        self.time = self.robot.get_time()
        self.lidar = self.robot.get_lidar_range_list()
        self.left_motor_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_motor_ticks = self.robot.get_right_motor_encoder_ticks()

        # EX03 stuff
        if self.lidar is None:
            print("Lidar data is NULL!")
            self.range_list = []
            return
        else:
            self.range_list = self.lidar

        if not self.range_list or not isinstance(self.range_list, list):
            print("Invalid or empty Lidar data, skipping sensing.")
            self.range_list = []
            return

        objects = []
        in_object = False
        start_idx = None

        for i in range(1, len(self.range_list)):
            if self.range_list[i] is None or self.range_list[i] == float('inf') or self.range_list[i - 1] is None or \
                    self.range_list[i - 1] == float('inf'):
                in_object = False
                continue

            if not in_object and self.range_list[i] < self.range_list[i - 1] * 0.8:
                in_object = True
                start_idx = i

            elif in_object and self.range_list[i] > self.range_list[start_idx] * 1.2:
                center_idx = round(start_idx + (i - start_idx) / 2)
                objects.append((self.range_list[center_idx], self._get_angle(center_idx)))
                in_object = False

        self.detected_objects = self._filter_objects(objects)

    def get_objects_range_list(self) -> list | None:
        """Return the detected objects range list.

        Based on the robot's lidar range list measurements, extract objects and
        return a list of detected objects. Each object contains the distance
        in meters and angle in radians in terms of the scan.

        The expected angle is the angle for the index that is the center of the
        object (floored).

        Example:
        For example, object exists at lidar range list indexes 7, 8, 9, 10.
        Then the angle should be the same as it was for index 8. The
        distance should also be the same as it was for index 8.

        Returns:
            list: A list of tuples, where each tuple represents an object with
            distance and angle [(distance, angle), (distance, angle), ...].
            None if no objects are detected.
        """
        return self.detected_objects if self.detected_objects else None

    def _get_angle(self, index):
        num_points = len(self.range_list)
        fov = 2 * math.pi
        angle_per_step = fov / num_points
        return index * angle_per_step

    def _filter_objects(self, objects):
        min_distance_threshold = 0.2
        valid_objects = []

        for obj in objects:
            if obj[0] > min_distance_threshold:
                valid_objects.append(obj)

        return valid_objects


    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        state_actions = {
            "init": self._handle_init,
            "search": self._handle_search,
            "turning_to_object": self._handle_turning,
            "approaching": self._handle_approaching,
            "fixing_trajectory": self._handle_fixing_trajectory,
            "finished": self._handle_finished
        }

        if self.state in state_actions:
            state_actions[self.state]()
    def _handle_init(self):
        print("HELLO, I ROBOT!")
        self.state = "search"

    def _handle_search(self):
        self.left_velocity = -0.5
        self.right_velocity = 0.5
        print("I, SEARCH")
        if self.detected_objects:
            self.state = "turning_to_object"
            print("I, FIND")

    def _handle_turning(self):
        if 4.0 < self.detected_objects[0][1] < 5.5:
            self.left_velocity = -0.1
            self.right_velocity = 0.1

        print("I, TURN")
        if 4.7 < self.detected_objects[0][1] < 4.75:
            self.state = "approaching"
            print("I, APPROACH")
        else:
            self.state = "search"
            print("I, GO BACK SEARCH")

    def _handle_approaching(self):
        self.left_velocity = 1
        self.right_velocity = 1
        if self.detected_objects:
            if 4.65 > self.detected_objects[0][1] > 4.8:
                self.state = "fixing_trajectory"
        if self.detected_objects and self.detected_objects[0][0] < 0.3:
            self.state = "finished"
            print("I, FINISHED")
        elif not self.detected_objects:
            self.state = "search"
            print("fked up situation")

    def _handle_fixing_trajectory(self):
        print("I, FIX")
        if self.detected_objects[0][1] < 4.65:
            print("I, LEFT")
            self.left_velocity = -0.1
            self.right_velocity = 0.1
        elif self.detected_objects[0][1] > 4.7:
            print("I, RIGHT")
            self.left_velocity = 0.1
            self.right_velocity = -0.1
        else:
            self.state = "approaching"

    def _handle_finished(self):
        self.left_velocity = 0
        self.right_velocity = 0
        print("I, END(myself)")

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