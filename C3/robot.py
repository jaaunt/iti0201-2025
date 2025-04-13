def plan(self) -> None:
    current_time = self.robot.get_time()

    # LIDAR scan
    front = self.lidar[470:490] if self.lidar else []
    left = self.lidar[400:470] if self.lidar else []
    right = self.lidar[490:560] if self.lidar else []

    min_front = min((d for d in front if d), default=1.0)
    min_left = min((d for d in left if d), default=1.0)
    min_right = min((d for d in right if d), default=1.0)
    obstacle_close = min_front < 0.3 or min_left < 0.35 or min_right < 0.35

    # Leave avoidance after push-forward
    if self.avoiding_obstacle and not obstacle_close:
        if current_time - self.avoid_start_time >= self.avoid_push_time:
            print("Obstacle cleared, resuming cube tracking")
            self.avoiding_obstacle = False

    # Enter avoidance
    if obstacle_close and not self.avoiding_obstacle:
        print("Obstacle detected, entering avoidance mode")
        self.avoiding_obstacle = True
        self.avoid_start_time = current_time
        self.state = "avoiding"

    if self.avoiding_obstacle:
        self.state = "avoiding"

    elif self.target_box:
        if abs(self.target_angle) > 0.1:
            if self.state != "adjusting":
                print("Adjusting to face cube")
            self.state = "adjusting"
        elif self.target_distance > 0.25:
            if self.state != "driving":
                print("Driving toward cube")
            self.state = "driving"
        else:
            if self.state != "arrived":
                print("Arrived at cube")
            self.state = "arrived"

    elif current_time - self.last_seen_time > 10:
        if self.state != "search":
            print("Searching for cube")
        self.state = "search"

    # Movement control
    if self.state == "adjusting":
        self.left_velocity = 0.3 if self.target_angle > 0 else -0.3
        self.right_velocity = -self.left_velocity

    elif self.state == "driving":
        self.left_velocity = 1.5
        self.right_velocity = 1.5

    elif self.state == "avoiding":
        if current_time - self.avoid_start_time < 0.8:
            if min_left < min_right:
                print("Avoiding: turning right")
                self.left_velocity = 1.2
                self.right_velocity = 0.4
            else:
                print("Avoiding: turning left")
                self.left_velocity = 0.4
                self.right_velocity = 1.2
        else:
            print("Avoiding: moving forward")
            self.left_velocity = 1.2
            self.right_velocity = 1.2

    elif self.state == "search":
        self.left_velocity = -0.5
        self.right_velocity = 0.5

    elif self.state == "arrived":
        self.left_velocity = 0
        self.right_velocity = 0
