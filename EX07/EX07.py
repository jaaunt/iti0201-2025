"""EX07: Object Detection."""
from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.blue_cubes = None

        self.image = None
        self.fov = None

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
                    for dy, dx in neighbours:
                        new_y, new_x = current_y + dy, current_x + dx
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

        blue_channel = self.image[:, :, 0]
        green_channel = self.image[:, :, 1]
        red_channel = self.image[:, :, 2]
        threshold = 50

        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)
        labled_mask, lable_count = self.find_blobs(mask)
        # labled_mask, lable_count = label(mask)  # testimis jaoks

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

        plt.imshow(self.image)
        for box in blobs:
            x_min, x_max, y_min, y_max = box
            plt.plot([x_min, x_max, x_max, x_min, x_min],
                     [y_min, y_min, y_max, y_max, y_min],
                     'r-', linewidth=5)
        # plt.axis("off")
        plt.show()

        return blobs if blobs else None

    def update_cube_objects(self):
        """Collect blue boxes to a list.

        Gets a list of bounding boxes and looks though them looking for objects that could be cubes.
        Adding them to a list and returning them.
        """
        boxes = self.get_object_bounding_box_list()
        if boxes is None:
            return None
        something = []
        for box in boxes:
            print(box)
            x_min = box[0]
            x_max = box[1]
            y_min = box[2]
            y_max = box[3]

            x_side = x_max - x_min
            y_side = y_max - y_min

            threshold = 20
            diffrence = 0
            if x_side > y_side:
                diffrence = x_side - y_side
            elif y_side > x_side:
                diffrence = y_side - x_side

            if diffrence <= threshold:
                something.append(box)
        return something

    def get_cube_objects(self) -> list | None:
        """Return the bounding boxes for detected objects.

        Image Data:
        - The image is of the form BGRA (height x width x 4).
        - The blue, green, and red channels correspond to the first, second, and third
          dimensions, respectively.
        - The alpha channel is included but not used for this detection.

        Bounding Box Data:
        - Each detected bounding box is returned as a tuple in the form
          `(x_min, x_max, y_min, y_max)`, where:
            - `x_min`: The minimum x-coordinate (left edge).
            - `x_max`: The maximum x-coordinate (right edge).
            - `y_min`: The minimum y-coordinate (top edge).
            - `y_max`: The maximum y-coordinate (bottom edge).

        Criteria for Detection:
        - Shape detection of correctly identifying the cube from the rest of objects.
        - Color detection of correctly identifying blue cubes from the rest of objects.

        Example:
          If an image contains a blue cube of size 100x100 pixels, the method will
          return a bounding box like:
          [(50, 150, 200, 300)]
          where:
            - The object spans from x=50 to x=150 horizontally.
            - The object spans from y=200 to y=300 vertically.

        Returns:
            [(x_min, x_max, y_min, y_max), ...]: A list of bounding boxes for
            detected cubes.
            Returns `None` if no cubes are detected.
        """
        self.blue_cubes = self.update_cube_objects()
        return self.blue_cubes if self.blue_cubes else None

    def sense(self) -> None:
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.image = self.robot.get_camera_rgb_image()
        self.update_cube_objects()

        self.fov = self.robot.get_camera_field_of_view()

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
