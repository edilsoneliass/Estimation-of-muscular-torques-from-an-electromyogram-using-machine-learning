import matplotlib.pyplot as plt
import numpy as np


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)


class Particle:
    def __init__(self, position=None, velocity=None):
        self.position = position
        self.velocity = velocity

    def update_position(self, dt):
        self.position += self.velocity * dt
        return self.position


def collision(particle, width, height):
    if particle.position.x <= 0 or particle.position.x >= width:
        particle.velocity.x = -particle.velocity.x
    if particle.position.y <= 0 or particle.position.y >= height:
        particle.velocity.y = -particle.velocity.y
    return particle


def end_trajectory(particle, width, height):
    vertex = [(0, 0), (0, height), (width, 0), (width, height)]
    if (particle.position.x, particle.position.y) in vertex:
        return True
    return False


def velocity_logic(width, height, vertex):
    if vertex[0] == 0 and vertex[1] == 0:
        return Vector(1, 1)
    elif vertex[0] == width and vertex[1] == 0:
        return Vector(-1, 1)
    elif vertex[0] == 0 and vertex[1] == height:
        return Vector(1, -1)
    else:
        return Vector(-1, -1)


def animation(width, height, vertex):
    n = max(width, height)
    fig, ax = plt.subplots()
    '''Plot das bordas do retângulo'''
    ax.plot([0, 0], [0, height], color='black')
    ax.plot([0, width], [height, height], color='black')
    ax.plot([width, width], [0, height], color='black')
    ax.plot([0, width], [0, 0], color='black')
    ax.set_xlim([0, n])
    ax.set_ylim([0, n])
    ax.set_yticks(range(0, n+1))
    ax.set_xticks(range(0, n+1))
    ax.grid(True)
    flag_trajectory = False
    particle = Particle()
    particle.position = Vector(vertex[0], vertex[1])
    particle.velocity = velocity_logic(width, height, vertex)
    previous_position = particle.position
    while not flag_trajectory:
        particle.update_position(dt=1)
        ax.plot([previous_position.x, particle.position.x], [previous_position.y, particle.position.y], color='red')
        if end_trajectory(particle, width, height):
            flag_trajectory = True
        collision(particle, width, height)
        previous_position = particle.position
        print(particle.position.x, particle.position.y)
    plt.show()
    return


animation(14, 5, [0, 0])



