import numpy as np


class Boid:
    def __init__(self):
        # TODO: further implement the agents, these ranges are just picked randomly
        self.pos_x = np.random.uniform(0, 1)
        self.pos_y = np.random.uniform(0, 1)
