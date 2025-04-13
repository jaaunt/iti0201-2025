"""C3."""
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.image = None
        self.fov = None
        self.lidar = []
        self.target_box = None
        self.left_velocity = 0
        self.right_velocity = 0
        self.state = "searching"
        self.forward_ticks_remaining = 0
        self.turning = False

    def sense(self) -> None:
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.target_box = self._find_blue_cube()

    def plan(self) -> None:
        if self.forward_ticks_remaining > 0:
            print(f"Forward detour... ticks left: {self.forward_ticks_remaining}")
            self.left_velocity = 2.0
            self.right_velocity = 2.0
            self.forward_ticks_remaining -= 1
            return

        if self.turning:
            print("Turning right 90 degrees")
            self.left_velocity = 2.0
            self.right_velocity = -2.0
            self.turning = False
            self.forward_ticks_remaining = 20  # Approx 2s forward
            return

        if self._object_in_front(threshold=0.4):
            print("Obstacle ahead < 0.4m – starting right turn + forward")
            self.turning = True
            return

        if self.target_box is None:
            print("No cube – spinning to search")
            self.left_velocity = -2.0
            self.right_velocity = 2.0
            return

        print("Cube found – driving forward")
        self.left_velocity = 2.5
        self.right_velocity = 2.5

    def act(self) -> None:
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self) -> None:
        self.sense()
        self.plan()
        self.act()

    def _find_blue_cube(self):
        if self.image is None:
            return None

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        threshold = 50

        mask = (blue > green + threshold) & (blue > red + threshold)
        labeled_mask, count = self._find_blobs(mask)

        if count == 0:
            return None

        for i in range(1, count + 1):
            pixels = np.column_stack(np.where(labeled_mask == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min

            if abs(width - height) < 20 and height < 150:
                return (x_min, x_max, y_min, y_max)

        return None

    def _object_in_front(self, threshold=0.4):
        if not self.lidar:
            return False

        center = len(self.lidar) // 2
        span = 20
        front_values = self.lidar[center - span:center + span + 1]

        for d in front_values:
            if d is not None and d != float("inf") and d < threshold:
                return True

        return False

    def _find_blobs(self, mask):
        height, width = mask.shape
        labeled = np.zeros_like(mask, dtype=np.uint32)
        label_id = 1
        stack = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y, x in np.argwhere(mask):
            if labeled[y, x] == 0:
                labeled[y, x] = label_id
                stack.append((y, x))
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = label_id
                                stack.append((ny, nx))
                label_id += 1

        return labeled, label_id - 1
