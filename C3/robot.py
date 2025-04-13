import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.image: np.ndarray | None = None
        self.fov: float | None = None
        self.lidar: list[float] = []
        self.state = "approaching"
        self.blue_cubes = None
        self.left_velocity = 0
        self.right_velocity = 0
        self.last_seen_cube_box = None
        self.last_cube_seen_time = 0.0
        self.time = 0.0
        self.spin_timer = 0.0

    def find_blobs(self, mask):
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
        if self.image is None:
            return None

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        threshold = 50

        mask = (blue > green + threshold) & (blue > red + threshold)
        labeled_mask, label_count = self.find_blobs(mask)

        if label_count == 0:
            return None

        blobs = []
        for i in range(1, label_count + 1):
            pixels = np.column_stack(np.where(labeled_mask == i))
            if pixels.size == 0:
                continue
            y_min, x_min = pixels.min(axis=0)
            y_max, x_max = pixels.max(axis=0)
            blobs.append((x_min, x_max, y_min, y_max))

        return blobs if blobs else None

    def update_cube_objects(self):
        boxes = self.get_object_bounding_box_list()
        if boxes is None:
            return None
        cubes = []
        for box in boxes:
            x_min, x_max = box[0], box[1]
            y_min, y_max = box[2], box[3]
            x_side = x_max - x_min
            y_side = y_max - y_min
            ratio = x_side / y_side if y_side != 0 else 0
            if 0.8 <= ratio <= 1.2:
                cubes.append(box)
        return cubes

    def get_cube_objects(self) -> list | None:
        self.blue_cubes = self.update_cube_objects()
        if self.blue_cubes:
            self.last_seen_cube_box = self.blue_cubes[0]
            self.last_cube_seen_time = self.time
        return self.blue_cubes if self.blue_cubes else None

    def get_cube_angle(self):
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

    def _get_front_distance(self):
        if not self.lidar:
            return float('inf')
        center_index = len(self.lidar) // 2
        front_values = self.lidar[center_index - 10:center_index + 10]
        valid = [d for d in front_values if d is not None and d != float('inf')]
        return min(valid) if valid else float('inf')

    def _handle_approaching(self):
        print("STATE: APPROACHING")
        angle = self.get_cube_angle()

        if angle is None:
            print("No cube seen – spinning to find")
            self.left_velocity = -2.0
            self.right_velocity = 2.0
            self.spin_timer += 1
            if self.spin_timer > 100:
                print("Cube not found after spinning – stopping")
                self.left_velocity = 0
                self.right_velocity = 0
        else:
            self.spin_timer = 0
            if abs(angle) < 0.1:
                print("Cube centered – switching to DRIVING")
                self.left_velocity = 0
                self.right_velocity = 0
                self.state = "driving"
            elif angle > 0:
                print("Cube to the right – turning right")
                self.left_velocity = 1.5
                self.right_velocity = -1.5
            else:
                print("Cube to the left – turning left")
                self.left_velocity = -1.5
                self.right_velocity = 1.5

    def _handle_driving(self):
        print("STATE: DRIVING")
        cube_boxes = self.get_cube_objects()
        front_distance = self._get_front_distance()

        if cube_boxes:
            if front_distance < 0.5:
                print("Obstacle ahead and cube still visible – avoiding")
                self.left_velocity = 2.0
                self.right_velocity = 0.5
            else:
                print("Cube visible and path clear – moving forward")
                self.left_velocity = 4.0
                self.right_velocity = 4.0
        else:
            time_since_seen = self.time - self.last_cube_seen_time
            if front_distance < 0.5:
                print("Cube not visible but obstacle ahead – STOPPED (probably arrived)")
                self.left_velocity = 0
                self.right_velocity = 0
            elif time_since_seen < 2.0:
                print("Cube just disappeared – likely occluded – keep going")
                self.left_velocity = 3.0
                self.right_velocity = 3.0
            else:
                print("Lost cube – returning to APPROACHING")
                self.state = "approaching"

    def sense(self):
        self.image = self.robot.get_camera_rgb_image()
        self.fov = self.robot.get_camera_field_of_view()
        self.lidar = self.robot.get_lidar_range_list()
        self.time = self.robot.get_time()
        self.update_cube_objects()

    def plan(self):
        if self.state == "approaching":
            self._handle_approaching()
        elif self.state == "driving":
            self._handle_driving()

    def act(self):
        self.robot.set_left_motor_velocity(self.left_velocity)
        self.robot.set_right_motor_velocity(self.right_velocity)

    def spin(self):
        self.sense()
        self.plan()
        self.act()
