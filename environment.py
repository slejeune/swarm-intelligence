import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from typing import Optional, Tuple, Union

from sklearn.neighbors import KDTree

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
        domain: np.ndarray = np.array([[-6, -6], [6, 6]]),
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
        self.spacing = spacing
        self.upper_straight = self.Straight(
            center_x=self.center_x,
            center_y=self.center_y + (self.curve_height/2 + self.straight_height / 2),
            width=self.straight_width,
            height=self.straight_height,
            upper=True,
        )
        self.lower_straight = self.Straight(
            center_x=self.center_x,
            center_y=self.center_y - (self.curve_height/2 + self.straight_height / 2),
            width=self.straight_width,
            height=self.straight_height,
            upper=False,
        )
        self.left_ellipsoid = self.Ellipsoid(
            center_x=self.center_x + self.straight_width / 2,
            center_y=self.center_y,
            a=spacing,
            b=curve_height/2,
            width=curve_width,
            height=straight_height,
            left=True,
        )
        self.right_ellipsoid = self.Ellipsoid(
            center_x=self.center_x + self.straight_width / 2,
            center_y=self.center_y,
            a=spacing,
            b=curve_height/2,
            width=curve_width,
            height=straight_height,
            left=False,
        )
        self.components = [self.upper_straight, self.lower_straight, self.left_ellipsoid, self.right_ellipsoid]
        self.boids = np.array([])
        self.domain = domain

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

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random position within the straight.
        Args:
            size(int): The number of positions to sample.
        Returns:
            Tuple[float, float]: The sampled x and y positions.
        """
        samples_x = np.zeros(size)
        samples_y = np.zeros(size)
        for i, component in enumerate(self.components):
            (
                samples_x[i * (size // len(self.components)) : (i + 1) * (size // len(self.components))],
                samples_y[i * (size // len(self.components)) : (i + 1) * (size // len(self.components))],
            ) = component.sample(size=size // len(self.components))
        remainder = size - len(self.components) * (size // len(self.components))
        if remainder > 0:
            samples_x[-remainder:], samples_y[-remainder:] = np.random.choice(self.components).sample(remainder)

        self.boids = np.array([Boid(samples_x[i], samples_y[i], domain=self.domain) for i in range(samples_x.shape[0])])

        return samples_x, samples_y

    def update(self) -> Tuple[list, list]:
        """
        Update the position of the boids each timestep.
        Returns:
            list: The new positions of the boids.
        """
        offsets = []
        markers = []
        self.align()
        self.steer_towards_track()

        for boid in self.boids:
            boid.update()

            offsets.append([boid.pos_x, boid.pos_y])
            marker = MarkerStyle(">")
            marker._transform = marker.get_transform().rotate_deg(np.degrees(boid.direction))
            markers.append(marker)

        return offsets, markers
    
    

    def align(self):
        boid_positions = np.array([[b.pos_x, b.pos_y] for b in self.boids])
        tree = KDTree(boid_positions, leaf_size=10)
        for boid in self.boids:
            neighbours = self.get_neighbours(boid, radius=0.5, tree=tree)
            directions = np.array([b.direction for b in neighbours])
            boid.direction = np.arctan2(np.mean(np.sin(directions)), np.mean(np.cos(directions)))

    def get_neighbours(self, boid: Boid, radius: float, tree: KDTree) -> np.ndarray:
        indices = tree.query_radius([[boid.pos_x, boid.pos_y]], r=radius, count_only=False, return_distance=False)
        return self.boids[indices[0]]

    def border(x,y,half_width,half_height,spacing):
        return np.where((-half_width < x) & (x < half_width), 0, (x-np.sign(x)*half_width)**2/spacing**2) + y**2/half_height**2
    
    def steer_towards_track(self):
        for boid in self.boids:
            if Track.border(boid.pos_x,boid.pos_y,self.straight_width/2,self.curve_height/2,self.spacing) < 1:
                boid.direction = np.angle(np.complex(boid.pos_x,boid.pos_y))

            if Track.border(boid.pos_x,boid.pos_y,self.straight_width/2,self.curve_height/2+self.straight_height,self.spacing+self.curve_width) > 1:
                boid.direction = np.angle(np.complex(-boid.pos_x,-boid.pos_y))

    class Straight:
        def __init__(self, center_x: float, center_y: float, width: float, height: float, upper: bool):
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
            self.upper = upper

        def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Sample a random position within the straight.
            Args:
                size(int): The number of positions to sample.
            Returns:
                Tuple[np.ndarray, np.ndarray]: The sampled x and y positions.
            """
            x = np.random.uniform(low=-0.5, high=0.5, size=size) * self.width + self.center_x
            y = np.random.uniform(low=-0.5, high=0.5, size=size) * self.height + self.center_y
            return x, y

        def contains(self, boid: Boid):
            """
            Verify whether the position of the boid falls within this component.
            Args:
                boid (Boid): The boid.
            Returns:
                bool: True if the position of the boid falls within this component.
                bool: True if the position of the boid is on the inside of the track instead of within the track
            """
            contains_x = self.center_x - self.width / 2 <= boid.pos_x <= self.center_x + self.width / 2
            contains_y = self.center_y - self.height / 2 <= boid.pos_y <= self.center_y + self.height / 2

            if self.upper:
                return contains_x and contains_y,  self.center_y - self.height / 2 > boid.pos_y

            return contains_x and contains_y, boid.pos_y > self.center_y + self.height / 2

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
            An ellipsoid can be described as x**2/a**2 + y**2/b**2 = 1.
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

        def inner_curve(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return self.center_x + np.sqrt(self.a ** 2 * (1 - y ** 2 / self.b ** 2))

        def outer_curve(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return self.center_x + np.sqrt((self.a + self.width) ** 2 * (1 - y ** 2 / (self.b + self.height) ** 2))

        def contains(self, boid: Boid):
            """
            Verify whether the position of the boid falls within this component.

            Args:
                boid (Boid): The boid.
            Returns:
                bool: True if the position of the boid falls within this component.
            """
            if not self.left:
                right_of_box = boid.pos_x > self.center_x
                larger_than_inner = (boid.pos_x - self.center_x) ** 2 / self.a ** 2 + (
                    boid.pos_y - self.center_y
                ) ** 2 / self.b ** 2 >= 1
                smaller_than_outer = (boid.pos_x - self.center_x) ** 2 / (self.a + self.width) ** 2 + (
                    boid.pos_y - self.center_y
                ) ** 2 / (self.b + self.height) ** 2 <= 1

                inside = not larger_than_inner

                return larger_than_inner and smaller_than_outer and right_of_box, inside

            left_of_box = boid.pos_x < -self.center_x
            larger_than_inner = (boid.pos_x + self.center_x) ** 2 / self.a ** 2 + (
                boid.pos_y + self.center_y
            ) ** 2 / self.b ** 2 >= 1
            smaller_than_outer = (boid.pos_x + self.center_x) ** 2 / (self.a + self.width) ** 2 + (
                boid.pos_y + self.center_y
            ) ** 2 / (self.b + self.height) ** 2 <= 1

            inside = not larger_than_inner

            return larger_than_inner and smaller_than_outer and left_of_box, inside

        def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Sample a random position within the ellipsoid.
            Args:
                size(int): The number of positions to sample.
            Returns:
                Tuple[np.ndarray, np.ndarray]: The sampled x and y positions.
            """
            y_inner = np.random.uniform(low=self.center_y - self.b, high=self.center_y + self.b, size=size)
            if self.left:
                x = np.empty(size)
                for i in range(size):
                    x[i] = np.random.uniform(
                        low=-(self.inner_curve(y=y_inner[i])), high=-self.outer_curve(y=y_inner[i])
                    )
                return x, y_inner

            x = np.empty(size)
            for i in range(size):
                x[i] = np.random.uniform(low=self.inner_curve(y=y_inner[i]), high=self.outer_curve(y=y_inner[i]))
            return x, y_inner

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
            curve = -self.inner_curve(y=y) if self.left else self.inner_curve(y=y)
            ax.plot(curve, y, color=color)

            # outer ellipsoid
            y = np.linspace(
                start=self.center_y - (self.b + self.height), stop=self.center_y + (self.b + self.height), num=1000
            )
            curve = -self.outer_curve(y=y) if self.left else self.outer_curve(y=y)
            ax.plot(curve, y, color=color)
            return ax
