"""EX01: SPA."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot

    def sense(self):
        """Collect data and sort it to variables."""
        pass

    def plan(self):
        """Analyse data and decide on an action."""
        pass

    def act(self):
        """Act on the plan."""
        pass

    def spin(self):
        """Spin."""
        self.sense()
        self.plan()
        self.act()
