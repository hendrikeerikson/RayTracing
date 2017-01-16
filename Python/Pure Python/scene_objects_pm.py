import numpy as np
import math


# normalize a numpy vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


class Sphere():
    def __init__(self, r, pos, color, id, transparent):
        # parameters of the sphere
        self.r = r
        self.c = pos

        # parameters for the shading model
        self.color = color  # RGB color of the sphere
        self.alpha = 20  # the shininess of the sphere, bigger number means smaller specular highlights
        self.diffuse = .5  # diffuse reflection constant , diffuse + specular can't be more than 1
        self.specular = .4  # specular reflection constant
        self.a_reflect = .5  # the amount of ambient light reflected 1 is all, 0 is none
        self.reflectiveness = 0.5
        self.transparent = transparent

        self.id = id

    def test_intersect(self, o, l):
        # returns the closest intersection point
        # solved with the sphere/line intersection equation
        a = np.dot(l, l)
        b = 2 * np.dot(l, o - self.c)
        c = np.dot(o - self.c, o - self.c) - self.r**2

        disc = b * b - 4 * a * c
        if disc >= 0:
            result_1 = (-b - disc**.5) / (2 * a)
            result_2 = (-b + disc**.5) / (2 * a)

            # return the closer point
            result = result_1
            if result_2 > result_1 > 0.0001:
                result = result_1
            elif result_2 > 0.0001:
                result = result_2
            else:
                result = np.inf

            is_inside = np.linalg.norm(o - self.c) < self.r + 0.0001

            return [result, is_inside]

        else:
            # if there are no intersection points return infinity
            return [np.inf, False]

    def get_normal(self, point):
        # return the normal vector of the sphere
        n = normalize(point - self.c)

        return n

    def get_diffuse(self, point):
        return self.diffuse * max(0.5, np.sign(math.cos(point[0] * 10)))


class Plane():
    def __init__(self, point, normal, color, id, transparent):
        # parameters for the shape
        self.p = point
        self.normal = normal

        # parameters for the shading model
        self.color = color  # RGB color of the sphere
        self.alpha = 5  # the shininess of the sphere, bigger number means smaller specular highlights
        self.diffuse = .4  # diffuse reflection constant, diffuse + specular can't be more than 1
        self.specular = 0.2  # specular reflection constant
        self.a_reflect = .4  # the amount of ambient light reflected 1 is all, 0 is none
        self.reflectiveness = 0
        self.transparent = transparent

        self.id = id

    def test_intersect(self, o, l):
        k = np.dot(l, self.normal)
        if k == 0:
            return [np.inf, False]

        p = np.dot((self.p - o), self.normal)
        if p == 0:
            return [np.inf, False]

        d = p / k

        if d < 1e-10:
            return [np.inf, False]

        else:
            return [d, False]

    def get_normal(self, point):
        return self.normal

    def get_diffuse(self, point):
        return self.diffuse * max(0.5, np.sign(math.cos(point[0] * 10)))
