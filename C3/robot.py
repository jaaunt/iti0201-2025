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
        self.avoid_timer = 0
        self.detour_timer = 0
        self.detour_direction = None

    def sense(self) -> None:
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.target_box = self._find_blue_cube()

    def plan(self) -> None:
        if self.state == "detour":
            self._handle_detour()
            return

        if self.state == "avoiding":
            self._handle_avoiding()
            return

        if self.target_box is None:
            print("No cube detected – rotating to search")
            self.left_velocity = -2.0
            self.right_velocity = 2.0
            self.state = "searching"
            return

        angle = self._get_cube_angle(self.target_box)
        distance = self._estimate_distance(self.target_box)
        print(f"Cube angle: {angle:.2f} rad, estimated distance: {distance:.2f} m")

        if self._object_in_front():
            print("Object detected in path – initiating DETOUR")
            self.state = "detour"
            self.detour_timer = 30  # ~2s depending on tick rate
            self.detour_direction = self._decide_detour_direction_by_camera()
            print(f"Detour direction: {self.detour_direction}")
            return

        if distance < 0.35:
            print("Cube reached – stopping.")
            self.left_velocity = 0
            self.right_velocity = 0
            return

        if abs(angle) > 0.2:
            print("Turning to align with cube")
            turn_speed = 1.5
            self.left_velocity = turn_speed if angle > 0 else -turn_speed
            self.right_velocity = -turn_speed if angle > 0 else turn_speed
            self.state = "approaching"
        else:
            print("Driving straight to cube")
            forward_speed = 2.5
            self.left_velocity = forward_speed
            self.right_velocity = forward_speed
            self.state = "approaching"

    def _handle_detour(self):
        print(f"DETOUR mode – {self.detour_timer} ticks remaining")

        if self.detour_timer > 20:
            # 90-degree turn
            if self.detour_direction == "left":
                self.left_velocity = -2.0
                self.right_velocity = 2.0
            else:
                self.left_velocity = 2.0
                self.right_velocity = -2.0
        else:
            # drive straight to bypass object
            self.left_velocity = 2.0
            self.right_velocity = 2.0

        self.detour_timer -= 1
        if self.detour_timer <= 0:
            print("Detour done – back to search")
            self.state = "searching"

    def _handle_avoiding(self):
        print(f"Avoiding... {self.avoid_timer} ticks left")

        left = self._sector_obstacle(-45, -10)
        right = self._sector_obstacle(10, 45)

        if left and not right:
            print("Obstacle on left – turning right")
            self.left_velocity = 2.0
            self.right_velocity = -1.0
        elif right and not left:
            print("Obstacle on right – turning left")
            self.left_velocity = -1.0
            self.right_velocity = 2.0
        else:
            print("Obstacle ahead – driving forward slightly")
            self.left_velocity = 2.0
            self.right_velocity = 2.0

        self.avoid_timer -= 1
        if self.avoid_timer <= 0:
            print("Avoiding complete – switching to search")
            self.state = "searching"

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

    def _object_in_front(self, threshold=0.4):
        return self._sector_obstacle(-20, 20, threshold)

    def _sector_obstacle(self, start_deg, end_deg, threshold=0.5):
        if not self.lidar:
            return False

        lidar_fov = np.pi
        num_rays = len(self.lidar)
        center = num_rays // 2

        start_rad = np.deg2rad(start_deg)
        end_rad = np.deg2rad(end_deg)
        start_index = int(center + (start_rad / lidar_fov) * num_rays)
        end_index = int(center + (end_rad / lidar_fov) * num_rays)

        start_index = max(0, min(num_rays - 1, start_index))
        end_index = max(0, min(num_rays - 1, end_index))

        for i in range(start_index, end_index + 1):
            d = self.lidar[i]
            if d is not None and d != float("inf") and d < threshold:
                return True

        return False

    def _decide_detour_direction_by_camera(self):
        if self.image is None:
            return "left"  # default

        h, w, _ = self.image.shape
        left_half = self.image[:, :w // 2]
        right_half = self.image[:, w // 2:]

        left_sum = left_half[:, :, 0].sum()
        right_sum = right_half[:, :, 0].sum()

        print(f"Camera analysis – Blue sum left: {left_sum}, right: {right_sum}")
        if left_sum > right_sum:
            return "right"
        else:
            return "left"

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
