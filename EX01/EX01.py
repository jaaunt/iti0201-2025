"""EX01: SPA."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.collected_data = "data"
        self.action = "spin"
        self.activity = ""

    def sense(self):
        """Collect data and sort it to variables."""
        return self.collected_data

    def plan(self):
        """Analyse data and decide on an action."""
        if self.action == "spin":
            self.activity = "spin"
        else:
            self.activity = "stop"

    def act(self):
        """Act on the plan."""
        self.plan()
        if self.activity == "stop":
            print("stop")
        elif self.activity == "spin":
            print("spin")
        else:
            print("nothing")