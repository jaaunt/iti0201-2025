"""EX06: Colors."""
from __future__ import annotations

import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
          robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.time = 0
        self.image = None
        self.fov = None
        self.object = []
        self.current_colour = None


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
        blue_channel =self.image[:, :, 0]
        green_channel =self.image[:, :, 1]
        red_channel =self.image[:, :, 2]
        threshold = 10
        mask = (blue_channel > green_channel) + threshold & (blue_channel > red_channel) + threshold

        labled_mask, lable_count = self.find_blobs(mask)
        blobs = {}
        for i in range(1, lable_count + 1):
            blobs[i] = np.column_stack(np.where(labled_mask == i))
        # x ja y osa ise
    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.time = self.robot.get_time()
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_params()

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
