import pyopencl as cl
import pyopencl.tools
from OpenGL.GL import *

import numpy as np
from Constants import *


# normalize a numpy vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


# read the kernel file, remember to include the file type suffix
def read_kernel(kernel_name):
        with open(kernel_name) as f:
            kernel_string = f.read()

        return kernel_string


def read_materials(device):
    # create a dtype to house a materials properties
    Material = np.dtype([("color", np.float32, (4,)),
                         ("diffuse", np.float32),
                         ("specular", np.float32),
                         ("ambient", np.float32),
                         ("shininess", np.float32),
                         ])

    f = open("materials.txt", "r")

    mat_list = []

    # read the materials into a list
    for i in f:
        if not i[0] == "#":
            components = i.split(":")
            rgb = components[0].split(";")

            rgb = map(int, rgb)
            rgb = map(lambda x: x/255, rgb)

            rgb_ar = np.zeros((4,), dtype=np.float32)
            rgb_ar[0:3] = list(rgb)

            diffuse = np.float32(float(components[1]))
            specular = np.float32(float(components[2]))
            ambient = np.float32(float(components[3]))
            shininess = np.float32(float(components[4]))

            mat_list += [[rgb_ar, diffuse, specular, ambient, shininess]]

    # create a list of empty structs
    mat_struct_list = np.empty(len(mat_list), Material)

    # fill the structs with data
    for x, i in enumerate(mat_list):
        mat_struct_list[x]["color"] = i[0]
        mat_struct_list[x]["diffuse"] = i[1]
        mat_struct_list[x]["specular"] = i[2]
        mat_struct_list[x]["ambient"] = i[3]
        mat_struct_list[x]["shininess"] = i[4]

    return mat_struct_list


def read_triangles(device):
    # create a dtype for a triangles data
    Triangle = np.dtype([("v0", np.float32, (4,)),
                         ("v1", np.float32, (4,)),
                         ("v2", np.float32, (4,)),
                         ("normal", np.float32, (4,)),
                         ("mat_index", np.uint32)])

    f = open("triangles.txt", "r")

    tri_list = []

    for i in f:
        if not i[0] == "#":
            triangle = []
            for j in range(3):
                components = i.split(":")
                v0 = components[0].split(";")

                v0 = map(int, v0)
                v0 = map(lambda x: x / 255, v0)

                v0_ar = np.zeros((4,), dtype=np.float32)
                v0_ar[0:3] = list(v0)

                triangle += [v0]

            normal = np.cross((triangle[1]-triangle[0]), (triangle[2]-triangle[0]))
            triangle += [normal]
            tri_list += [triangle]
            triangle = []

    tri_struct_list = np.empty(len(tri_list), Triangle)

    for x, i in enumerate(tri_list):
        tri_struct_list[x]["color"] = i[0]
        tri_struct_list[x]["diffuse"] = i[1]
        tri_struct_list[x]["specular"] = i[2]

    return tri_struct_list


# OpenCL initialization
def cl_init():
    # the platfrom has to be manually selected to be AMD or NVIDIA
    platform = cl.get_platforms()[0]

    # select the first device that is a GPU
    device = platform.get_devices()
    context = cl.Context(device,
                         properties=[(cl.context_properties.PLATFORM, platform)])

    queue = cl.CommandQueue(context)

    print(device[0].extensions)
    print(device[0].vendor)

    return context, queue, device[0]


# takes a string in fromat raw RGBA
def texture_from_array(array, shape):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, shape[1], shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, array)

    return texture


def create_scene_buffer(device):
    Sphere = np.dtype([("radius", np.float32),
                       ("pos", np.float32),
                       ("mat_index", np.uint32)])

    Sphere = cl.tools.match_dtype_to_c_struct(device, "Sphere", Sphere)
    Sphere = cl.tools.get_or_register_dtype("Sphere", Sphere)

    Light = np.dtype([("pos", np.float32),
                         ("ambient", np.float32),
                         ("specular", np.float32),
                         ("diffuse", np.float32)])

    Light = cl.tools.match_dtype_to_c_struct(device, "Light", Light)
    Light = cl.tools.get_or_register_dtype("Light", Light)


def create_camera_struct(width, height, device):
    # width and height of the virtual screen
    size_x = camera_size
    size_y = camera_size * (height / width)

    # world coordinates of the middle of the virtual screen
    camera_middle = camera_pos + camera_dir * camera_dist

    # use this matrix to get the vector that points from the screen's center to it's rigth
    rot_mat1 = np.matrix([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])

    # compute the two vectors used to get the point of each pixel on the screen
    dir_left = normalize(np.transpose(camera_dir * rot_mat1))
    dir_left = dir_left.reshape(-1)
    dir_left *= (size_x / width)
    dir_left *= -1
    dir_up = normalize(np.cross(camera_dir, dir_left))
    dir_up *= (size_y / height)

    # create the buffers
    camera = np.zeros((4,), dtype=np.float32)
    camera_middle_2 = np.zeros((4,), dtype=np.float32)
    dir_left_2 = np.zeros((4,), dtype=np.float32)
    dir_up_2 = np.zeros((4,), dtype=np.float32)

    camera[0:3] = camera_pos
    camera_middle_2[0:3] = camera_middle
    dir_left_2[0:3] = dir_left
    dir_up_2[0:3] = dir_up
    w_h = np.array([width, height], dtype=np.int)

    Camera = np.dtype([("camera_pos", np.float32, (4,)),
                       ("camera_middle", np.float32, (4,)),
                       ("camera_up", np.float32, (4,)),
                       ("camera_left", np.float32, (4,)),
                       ("w_h", np.uint32, (2,))])

    camera_struct = np.empty(1, Camera)
    camera_struct[0]["camera_pos"] = camera
    camera_struct[0]["camera_middle"] = camera_middle_2
    camera_struct[0]["camera_up"] = dir_up_2
    camera_struct[0]["camera_left"] = dir_left_2
    camera_struct[0]["w_h"] = w_h

    return camera_struct
