import numpy as np


# normalize a numpy vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


# GLOBAL VARIABLES
# set these before execution

camera_pos = np.array([0, 0, 0], dtype=np.float32)
camera_dist = 1.5  # this is the distance from the camera pos to the virtual screen
camera_size = 1  # this is the width of the virtual screen in world coordinates
camera_dir = np.array([0, 1, 0], dtype=np.float32)  # direction that the camera is facing

camera_dir = normalize(camera_dir)
