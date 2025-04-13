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

    def sense(self) -> None:
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.target_box = self._find_blue_cube()

    def plan(self) -> None:
        if self.target_box is None:
            print("No cube detected – rotating to search")
            self.left_velocity = -2.0
            self.right_velocity = 2.0
            return

        angle = self._get_cube_angle(self.target_box)
        distance = self._estimate_distance(self.target_box)

        print(f"Cube angle: {angle:.2f} rad, estimated distance: {distance:.2f} m")

        if distance < 0.35:
            print("Cube reached – stopping.")
            self.left_velocity = 0
            self.right_velocity = 0
            return

        if abs(angle) > 0.2:
            turn_speed = 1.5
            print("Turning to align with cube")
            self.left_velocity = turn_speed if angle > 0 else -turn_speed
            self.right_velocity = -turn_speed if angle > 0 else turn_speed
        else:
            if not self._is_path_clear_to_cube(angle):
                print("Obstacle in path to cube – stopping")
                self.left_velocity = 0
                self.right_velocity = 0
            else:
                forward_speed = 2.5
                print("Path is clear – driving toward cube")
                self.left_velocity = forward_speed
                self.right_velocity = forward_speed

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

            max_size = 150  # välista suured objektid nagu postid
            if abs(width - height) < 20 and height < max_size:
                return (x_min, x_max, y_min, y_max)

        return None

    def _get_cube_angle(self, box):
        x_min, x_max, _, _ = box
        width = self.image.shape[1]
        x_center = (x_min + x_max) / 2
        angle = ((x_center - width / 2) / (width / 2)) * (self.fov / 2)
        return angle

    def _estimate_distance(self, box):
        _, _, y_min, y_max = box
        pixel_height = y_max - y_min
        return 100.0 / pixel_height if pixel_height > 0 else float("inf")

    def _is_path_clear_to_cube(self, angle):
        if not self.lidar or self.fov is None:
            return True

        width = len(self.lidar)
        lidar_fov = np.pi
        lidar_angle_per_index = lidar_fov / width
        center_index = width // 2
        angle_index = int(angle / lidar_angle_per_index)
        index = center_index + angle_index

        check_range = 10
        indices_to_check = range(max(0, index - check_range), min(width, index + check_range + 1))

        for i in indices_to_check:
            value = self.lidar[i]
            if value is not None and value < 0.5:
                print(f"[WARNING] Obstacle detected at LIDAR[{i}] = {value:.2f}")
                return False

        return True

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
