import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from agents import Boid


class Track:
    def __init__(
        self,
        center_x: Optional[float] = 0,
        center_y: Optional[float] = 0,
        straight_width: Optional[float] = 4,
        straight_height: Optional[float] = 0.8,
        curve_width: Optional[float] = 1.2,
        curve_height: Optional[float] = 2.9,
        spacing: Optional[float] = 1.5,
    ):
        """
        Create a track. The track is subdivided into 4 components: 2 straight lanes and 2 ellipsoids.

        Args:
            center_x (Optional[float]): the x-coordinate of the center of the track.
            center_y (Optional[float]): the x-coordinate of the center of the track.
            straight_width (Optional[float]): The width (horizontal part) of the straight part of the track.
            straight_height (Optional[float]): The height (vertical part) of the straight part of the track.
            curve_width (Optional[float]): The width (space between two lines) of the ellipsoid.
            curve_height (Optional[float]): the height (radius-like) of the ellipsoid.
            spacing (Optional[float]): The offset of the inner ellipsoid with respect to the straight lanes.
        """
        self.center_x = center_x
        self.center_y = center_y
        self.straight_width = straight_width
        self.straight_height = straight_height
        self.curve_width = curve_width
        self.curve_height = curve_height
        self.upper_straight = self.Straight(
            center_x=self.center_x,
            center_y=self.center_y + (self.curve_height + self.straight_height / 2),
            width=self.straight_width,
            height=self.straight_height,
        )
        self.lower_straight = self.Straight(
            center_x=self.center_x,
            center_y=self.center_y - (self.curve_height + self.straight_height / 2),
            width=self.straight_width,
            height=self.straight_height,
        )
        self.left_ellipsoid = self.Ellipsoid(
            center_x=self.center_x + self.straight_width / 2,
            center_y=self.center_y,
            a=spacing,
            b=curve_height,
            width=curve_width,
            height=straight_height,
            left=True,
        )
        self.right_ellipsoid = self.Ellipsoid(
            center_x=self.center_x + self.straight_width / 2,
            center_y=self.center_y,
            a=spacing,
            b=curve_height,
            width=curve_width,
            height=straight_height,
            left=False,
        )
        self.components = [self.upper_straight, self.lower_straight, self.left_ellipsoid, self.right_ellipsoid]

    def plot(self, ax: plt.Axes, color: Optional[str] = "tab:blue") -> plt.Axes:
        """
        Plot the track.

        Args:
            ax (plt.Axes): The ax object whereon to plot.
            color (str): The color of the plotted track.

        Returns:
            (plt.Axes): The ax object whereon to plot.
        """
        for c in self.components:
            c.plot(ax=ax, color=color)
        return ax

    class Straight:
        def __init__(self, center_x: float, center_y: float, width: float, height: float):
            """
            Create a straight lane.

            Args:
                center_x (float): the x-coordinate of the center of the lane.
                center_y (float): the x-coordinate of the center of the lane.
                width (float): The width (horizontal part) lane.
                height (float): The height (vertical part) lane.
            """
            self.center_x = center_x
            self.center_y = center_y
            self.width = width
            self.height = height

        def contains(self, boid: Boid):
            """
            Verify whether the position of the boid falls within this component.
            Args:
                boid (Boid): The boid.

            Returns:
                bool: True if the position of the boid falls within this component.
            """
            contains_x = self.center_x - self.width <= boid.pos_x <= self.center_x + self.width
            contains_y = self.center_y - self.height <= boid.pos_y <= self.center_y + self.height
            return contains_x and contains_y

        def plot(self, ax: plt.Axes, color: str) -> plt.Axes:
            """
            Plot the ellipsoid.

            Args:
                ax (plt.Axes): The ax object whereon to plot.
                color (str): The color of the plotted track.

            Returns:
                (plt.Axes): The ax object whereon to plot.
            """
            # lower line
            ax.plot(
                [self.center_x - self.width / 2, self.center_x + self.width / 2],
                [self.center_y - self.height / 2, self.center_y - self.height / 2],
                color=color,
            )
            # upper line
            ax.plot(
                [self.center_x - self.width / 2, self.center_x + self.width / 2],
                [self.center_y + self.height / 2, self.center_y + self.height / 2],
                color=color,
            )
            return ax

    class Ellipsoid:
        def __init__(
            self, center_x: float, center_y: float, a: float, b: float, width: float, height: float, left: bool
        ):
            """
            Create a curved tack component.

            Args:
                center_x (float): the x-coordinate of the center of the ellipsoid.
                center_y (float): the x-coordinate of the center of the ellipsoid.
                a (float): ellipsoid parameter regulating the width.
                b (float): ellipsoid parameter regulating the position of the optimum of the ellipsoid.
                width (float): the spacing between the ellipsoid curves at the optimum.
                height (float): the spacing between the ellipsoid curves at the start
                    (should match with the height of the straight lane).
                left (bool): True if this is a left ellipsoid.
            """
            self.center_x = center_x
            self.center_y = center_y
            self.a = a
            self.b = b
            self.width = width
            self.height = height
            self.left = left

        def contains(self, boid: Boid):
            """
            Verify whether the position of the boid falls within this component.
            Args:
                boid (Boid): The boid.

            Returns:
                bool: True if the position of the boid falls within this component.
            """
            larger_than_inner = boid.pos_x ** 2 / self.a ** 2 + boid.pos_y ** 2 / self.b ** 2 >= 1
            smaller_than_outer = (
                boid.pos_x ** 2 / (self.a + self.width) ** 2 + boid.pos_y ** 2 / (self.b + self.height) ** 2 <= 1
            )
            return larger_than_inner and smaller_than_outer

        def plot(self, ax: plt.Axes, color: str) -> plt.Axes:
            """
            Plot the ellipsoid.

            Args:
                ax (plt.Axes): The ax object whereon to plot.
                color (str): The color of the plotted track.

            Returns:
                (plt.Axes): The ax object whereon to plot.
            """
            # inner ellipsoid
            y = np.linspace(start=self.center_y - self.b, stop=self.center_y + self.b, num=1000)
            curve = self.center_x + np.sqrt(self.a ** 2 * (1 - y ** 2 / self.b ** 2))
            curve = -curve if self.left else curve
            ax.plot(curve, y, color=color)

            # outer ellipsoid
            y = np.linspace(
                start=self.center_y - (self.b + self.height), stop=self.center_y + (self.b + self.height), num=1000
            )
            curve = self.center_x + np.sqrt((self.a + self.width) ** 2 * (1 - y ** 2 / (self.b + self.height) ** 2))
            curve = -curve if self.left else curve
            ax.plot(curve, y, color=color)
            return ax
