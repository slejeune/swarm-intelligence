import numpy as np
import matplotlib.pyplot as plt
import random


class Boid:

    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
    
    def update(self):

        # Add noise
        self.pos_x += random.uniform(-0.1, 0.1)
        self.pos_y += random.uniform(-0.1, 0.1)

        return
