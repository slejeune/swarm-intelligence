import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np


class Boid:

    def __init__(self, pos_x: int, pos_y: int, domain: np.ndarray, noise: float = 0.005):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.noise = noise
        self.direction = np.random.uniform(low=0, high=2 * np.pi)
        self.speed = 0.02  # np.random.uniform(low=0, high=0.05)
        self.domain = domain

    def update(self) -> None:
        # update position based on directional angle and speed
        self.pos_x += np.cos(self.direction) * self.speed
        self.pos_y += np.sin(self.direction) * self.speed

        # Add noise
        self.pos_x += random.uniform(-self.noise, self.noise)
        self.pos_y += random.uniform(-self.noise, self.noise)

        self.wrap()

    def wrap(self):
        width = self.domain[1, 0] - self.domain[0, 0]
        height = self.domain[1, 1] - self.domain[0, 1]
        if self.pos_x > self.domain[1, 0]:
            self.pos_x -= width
        if self.pos_x < self.domain[0, 0]:
            self.pos_x += width
        if self.pos_y > self.domain[1, 1]:
            self.pos_y -= height
        if self.pos_y < self.domain[0, 1]:
            self.pos_y += height
