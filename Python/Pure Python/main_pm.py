import numpy as np
import pygame
import sys
import scene_objects_pm
import multiprocessing as mp
from multiprocessing import managers as m
import scipy.spatial
from math import sin, cos, acos
from random import uniform

# coordinates x, y, z or width, depth, height
camera_focus = np.array([0., -2., .5])  # location of the camera focus
camera_dir = np.array([0., 1., 0.])  # normal vector for the viewpoint
camera_size_x, camera_size_y = 2, 1.5  # size of the imaginary screen
camera_dist = 1.5  # the screen distance from the focus

# using the Phong illumination model
ambient_light = np.array([0.1, 0.1, 0.1])  # strength of the ambient lighting

# light_source = np.array([-3., 3.5, 2])  # location of the light source
light_source = np.array([-2., 5., 3.])  # location of the light source

l_specular = np.array([0.5, 0.5, 0.5])  # color of specular highlights
l_diffuse = np.array([0.5, 0.5, 0.5])  # color of diffuse light

gamma = 1

ref_coefficient = 0.04  # reflection coefficient for Schlick's approximation, index of refraction is 1.5
glass = 1.52
air = 1.0001

photon_num = 100
photon_map_dict = {}
max_depth = 3  # maximum recursion depth
exposure = 40
find_radius = 1  # distance within photons are gathered

caustics_photon_num = 100
find_r_caustics = 0.1
find_n_caustics = 20
caustics_dict = {}

# scene objects in a dictionary, the keys are not important
'''
scene = {
    1: scene_objects_pm.Sphere(1.5, np.array([0., 4, 2.]), np.array([1., 0., 0.]), 1, True),
    2: scene_objects_pm.Plane(np.array([0., 5., 0.]), np.array([0., -1., 0.]), np.array([0.5, 0.5, 1.]), 8, False),

}
'''

scene = {
    1: scene_objects_pm.Sphere(1.5, np.array([-1.5, 4, 0.]), np.array([0., 1., 0.]), 2, True),
    2: scene_objects_pm.Sphere(1.5, np.array([2., 6, 0.]), np.array([1., 0., 0.]), 1, False),
    3: scene_objects_pm.Plane(np.array([0., 0., -1.5]), np.array([0., 0., 1.]), np.array([1., 1., 1.]), 3, False),
    4: scene_objects_pm.Plane(np.array([0., 8., 4.]), np.array([0., -1, 0]), np.array([0., 1., 0.]), 4, False),
    5: scene_objects_pm.Plane(np.array([4., 0., 0.]), np.array([-1., 0., 0.]), np.array([1., 0., 0.]), 5, False),
    6: scene_objects_pm.Plane(np.array([-4., 0., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.]), 6, False),
    7: scene_objects_pm.Plane(np.array([0., 0., 4.]), np.array([0., 0., -1.]), np.array([1., 1., 1.]), 7, False),
    8: scene_objects_pm.Plane(np.array([0., -2., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 0.]), 8, False),
}


# calculate the three corners of the camera screen (currently used only once, useful for making a moving camera)
def camera_corners():

    # a########b
    # ##########  <= this is the camera screen
    # c#########

    # because the camera can only be rotated in 2 dimensions
    # the third dimension is removed and then added back after the calculation
    camera_focus_2D = np.array([camera_focus[0], camera_focus[1]])
    camera_dir_2D = np.array([camera_dir[0], camera_dir[1]])

    # rotate camera_dir vector by 90 degrees to both sides using rotation matrices
    camera_dir_2D_270 = camera_dir_2D * np.matrix('0 1; -1 0')
    camera_dir_2D_90 = camera_dir_2D * np.matrix('0 -1; 1 0')

    # turn matrices back into arrays
    camera_dir_2D_270 = np.asarray(camera_dir_2D_270).reshape(-1)
    camera_dir_2D_90 = np.asarray(camera_dir_2D_90).reshape(-1)

    a_2D = (camera_focus_2D + camera_dir_2D * camera_dist) + camera_dir_2D_270 * (0.5*camera_size_x)
    b_2D = (camera_focus_2D + camera_dir_2D * camera_dist) + camera_dir_2D_90 * (0.5*camera_size_x)

    a = np.array([a_2D[0], a_2D[1], camera_focus[2] + 0.5 * camera_size_y])
    b = np.array([b_2D[0], b_2D[1], camera_focus[2] + 0.5 * camera_size_y])
    c = np.array([a_2D[0], a_2D[1], camera_focus[2] - 0.5 * camera_size_y])

    return [a, b, c]


# normalize a numpy vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


# create array of coordinates for rays on x and y axis
def create_screen_array_xy(a, b):
    # because the screen can rotate left and right the array is x and y coordinates
    # these coordinates are on the line that forms between the left and right edges of the screen
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])

    dist_x = abs(a[0] - b[0])  # might now work with negative coordinates
    dist_y = abs(a[1] - b[1])

    step_x = dist_x / screen_w
    step_y = dist_y / screen_w

    results = [a]

    for i in range(1, screen_w):
        results.append([results[-1][0] + step_x, results[-1][1] + step_y])

    return results


# create array of coordinates for rays on z axis
def create_screen_array_z(a, c):
    # because the camera can't rotate in the z axis, the array is only coordinates on the z axis
    a = a[2]
    c = c[2]

    dist_z = abs(a - c)
    step_z = dist_z / screen_h

    results = [a]

    for i in range(1, screen_h):
        results.append(results[-1] - step_z)

    return results


# calculate a reflected light ray, input is the negative of the light ray
def get_reflected_ray(r_in, norm):
    r_out = 2 * norm * (np.dot(r_in, norm)) - r_in
    return normalize(r_out)


# get ray from current point to the light
def get_light_ray(point):
    l_ray = (light_source - point)
    mag = np.sqrt(l_ray.dot(l_ray))
    return [normalize(l_ray), mag]


# get transmitted ray from current point with incident ray r_in
# this assumes the material is glass
# in_glass is a boolean to determine what material the origin of the ray is in
def get_transmitted_ray(r_in, norm, in_glass):
    if in_glass:
        n = glass/air
    else:
        n = air/glass

    dot = np.dot(norm, r_in)

    out_ray = -n * r_in + (n*dot - (1 - n**2 * (1 - dot**2))**0.5) * norm

    return normalize(out_ray)


# shoot photons from the light source
def shoot_photons():
    for n in range(0, photon_num):
        # the direction of the photon is random
        # photons are shot over the surface of a sphere
        u = uniform(0, 1)
        v = uniform(0, 1)

        theta = 2 * 3.1415 * u
        phi = acos(2*v - 1)

        # photon direction
        d = normalize(np.array([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)]))
        o = light_source  # photons originate from the light source
        power = l_specular  # place holder, color of photons when emitted

        # search for the first intersection and save the photon in a dictionary
        trace_photon(o, d, power, 0)

    # create a 3 dimensional tree of the photon map
    # this photon map doesn't contain information about the color of the photon or intersection angle
    data = list(photon_map_dict.keys())

    balanced_photon_map_new = scipy.spatial.cKDTree(data)

    return balanced_photon_map_new


# test all intersections and return the closest
def trace_photon(o, d, power, depth):
    if depth <= max_depth:  # guard that the recursion doesn't go on forever
        intersected = {}  # store all intersections here

        for i, x in enumerate(scene):
            dist = scene[x].test_intersect(o, d)[0]
            if dist != np.inf and dist > 0.01:
                intersected[dist] = scene[x]  # if an intersection was found, store the hit

        if len(intersected) > 0:
            dist_closest = sorted(intersected)[0]  # get the closest intersection distance
            body = intersected[dist_closest]  # intersection object
            point = o + d * dist_closest  # intersection point

            photon_map_dict[tuple(point)] = [d, power, body.id]

            power = body.color * 0.7

            trace_photon(point, get_reflected_ray(-d, body.get_normal(point)), power, depth + 1)


# create rays for caustics
def shoot_caustics():
    for n in range(0, caustics_photon_num):
        # the direction of the photon is random
        u = uniform(0, 1)
        v = uniform(0, 1)

        theta = 2 * 3.1415 * u
        phi = acos(2*v - 1)

        d = normalize(np.array([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)]))
        o = light_source
        power = np.array([.01, .01, .01])

        trace_caustics(o, d, power, False)

    data = list(caustics_dict.keys())

    balanced_caustics_map = scipy.spatial.cKDTree(data)

    return balanced_caustics_map


# trace photon for caustics
def trace_caustics(o, d, power, valid_caustics):
    intersected = {}  # store all intersections here

    for i, x in enumerate(scene):
        result = scene[x].test_intersect(o, d)

        if result[0] != np.inf and result[0] > 0.001:
            intersected[result[0]] = [scene[x], result[1]]  # if an intersection was found, store the hit

    if len(intersected) > 0:
        dist_closest = sorted(intersected)[0]  # get the closest intersection distance
        body = intersected[dist_closest][0]  # intersection object
        point = o + d * dist_closest  # intersection point
        in_material = intersected[dist_closest][1]
        normal = body.get_normal(point)

        if valid_caustics:
            caustics_dict[tuple(point)] = [d, power, body.id]

        if in_material and body.transparent:
            # caustics_dict[tuple(point)] = [d, power, body.id]
            trace_caustics(point, get_transmitted_ray(-d, -normal, True), power, True)

        elif body.transparent:
            trace_caustics(point, get_transmitted_ray(-d, normal, False), power, False)


# gather the photons in the gathering area
def gather_caustics(point, caustics_map_kd_tree):
    photon_dists = []  # store the distance of the photon from the current point
    closest_photons = []

    # get the closest photon to the current point, if no photon is nearby, returns nothing
    kd_tree_return = caustics_map_kd_tree.query(point, find_n_caustics, p=2, distance_upper_bound=np.inf)

    if len(kd_tree_return) == 0:
        return [[]]

    # get the photon coordinates from the photon map
    for x, i in enumerate(kd_tree_return[1]):
        closest_photons.append(caustics_map_kd_tree.data[i-1])
        photon_dists.append(kd_tree_return[0][x])

    return [closest_photons, photon_dists]


# gather the photons in the gathering area
def gather_photons(point, photon_map_kd_tree):
    photon_dists = []  # store the distance of the photon from the current point
    closest_photons = []

    # get the closest photon to the current point, if no photon is nearby, returns nothing
    kd_tree_return = photon_map_kd_tree.query_ball_point(point, find_radius, p=2)

    if len(kd_tree_return) == 0:
        return [[]]

    # get the photon coordinates from the photon map
    for x, i in enumerate(kd_tree_return):
        closest_photons.append(photon_map_kd_tree.data[i])
        photon_dists.append(np.linalg.norm(photon_map_kd_tree.data[i] - point))

    return [closest_photons, photon_dists]


# return true if there is and object between the current point and the light source
def calculate_shadow(o, l, body, dist_light):
    # test for intersection and stop after finding the first true collision

    for i, x in enumerate(scene):
        if scene[x] != body:
            dist = scene[x].test_intersect(o, l)[0]

            if dist != np.inf and abs(dist) > 1e-10 and dist <= dist_light:
                if not scene[x].transparent:
                    return True

    # if no objects intersect the ray, return false
    return False


# test all objects for ray intersection and then calculate color
def ray_trace(o, l, depth):

    intersected = {}

    depth += 1

    # test every object in scene
    for i, x in enumerate(scene):
        result = scene[x].test_intersect(o, l)
        dist = result[0]

        if dist != np.inf and dist > 0.005:
            # if intersection was found and is valid, add object and distance to it into a dictionary
            intersected[dist] = [scene[x], result[1]]

    if len(intersected) != 0:
        t_closest = sorted(intersected)[0]  # find the smallest intersection distance
        body = intersected[t_closest][0]  # find the object at that intersection distance from the dictionary
        is_in_material = intersected[t_closest][1]  # if the ray originates from inside a sphere
        point = o + l*t_closest  # world coordinates of the collision point
        normal = body.get_normal(point)  # calculate the surface normal of that object

        # if the second intersection is not the same material as the first one than the ray must have originated inside
        # the second material, thus the normal should be reversed
        if is_in_material:
            trans_vec = get_transmitted_ray(-l, -normal, True)
            n_point = point + 0.005 * normal
            return ray_trace(n_point, trans_vec, depth)

        light_ray, dist_light = get_light_ray(point)  # calculate the ray between current point and the light source
        ref_vec = get_reflected_ray(light_ray, normal)  # calculate the reflected ray

        trans_vec = get_transmitted_ray(-l, normal, False)


        # compute the components of the reflection model
        ambient = body.a_reflect * ambient_light
        '''

        # caustics
        photon_data = gather_caustics(point, photon_map_mp.getTreeCaustics())
        caustics = np.array([0., 0., 0.])

        if len(photon_data[0]) == 0:
            pass
        else:
            energy = []

            for x, i in enumerate(photon_data[0]):
                if photon_data[1][x] != np.inf:
                    light_color = photon_map_mp.getHashmapCaustics()[tuple(i)][1]
                    light_d = photon_map_mp.getHashmapCaustics()[tuple(i)][0]

                    weight = body.diffuse * max(0, (np.dot(-light_d, normal))) * l_diffuse
                    weight *= (1 - photon_data[1][x])

                    energy.append(light_color * weight)

            q = sorted([x for x in photon_data[1] if x != np.inf])

            if len(q) > 0:
                area = q[-1]**2 * 3.1416
                caustics = sum(energy) / area

            else:
                caustics = np.array([0, 0, 0])

        # regular photons
        photon_data = gather_photons(point, photon_map_mp.getKDtree())
        ambient = np.array([0., 0., 0.])

        if len(photon_data[0]) == 0:
            pass
        else:
            energy = []

            for x, i in enumerate(photon_data[0]):
                if photon_data[1][x] != np.inf:
                    light_color = photon_map_mp.getHashmap()[tuple(i)][1]
                    light_d = photon_map_mp.getHashmap()[tuple(i)][0]

                    weight = body.diffuse * max(0, (np.dot(-light_d, normal))) * l_diffuse
                    weight *= ((1 - photon_data[1][x]) / exposure)

                    energy.append(light_color * weight)

            q = sorted([x for x in photon_data[1] if x != np.inf])

            if len(q) > 0:
                area = q[-1]**2 * 3.1416
                ambient = sum(energy) / area

            else:
                ambient = np.array([0, 0, 0])
        '''
        specular = body.specular * max(0, (np.dot(ref_vec, -l))) ** body.alpha
        diffuse = body.diffuse * max(0, np.dot(normal, (light_ray-point)))

        # because light is additive, we add all the components together
        color = (diffuse + specular + 0.1)*body.color
        # color = (ambient + specular * 0.4) * body.color + caustics

        if body.reflectiveness > 0 and depth < 2:
            reflection_amount = ref_coefficient + (1 - ref_coefficient) * (1 - np.dot(normal, -l))**5

            if body.transparent:
                n_point = point - 0.001 * normal
                transmission = ray_trace(n_point, trans_vec, depth)
                reflection = ray_trace(point, (get_reflected_ray(-l, normal)), depth)

                a = color * 0.4
                b = transmission * (1-reflection_amount) * (1 - body.reflectiveness)
                c = reflection * reflection_amount * body.reflectiveness

                color = a + b + c
            else:
                reflection = ray_trace(point, (get_reflected_ray(-l, normal)), depth)
                color = color * (1-reflection_amount) * (1 - body.reflectiveness) + reflection * reflection_amount * body.reflectiveness

        # in some cases the light level is higher than the maximum of 255
        # clip the rgb components to fit between 0 and 255
        color = np.clip(color, 0, 1)
    else:
        color = np.array([0, 0, 0])

    return color


# simultaneously compute the rays
def start_tracing(ray_pixel_data):
    i, xy, j, z = ray_pixel_data

    P = np.array([xy[0], xy[1], z])  # current pixel coordinate in world coordinates
    D = np.array([P[0] - camera_focus[0], P[1] - camera_focus[1], P[2] - camera_focus[2]])  # direction of ray
    D = normalize(D)  # vector normalized

    depth = 0  # recursion depth

    pixel_color = ray_trace(camera_focus, D, depth)  # get the color of the current pixel

    return [i, j, pixel_color]


# ray tracing the scene
def draw_scene():
    # get the three corners of the screen
    camera_lens_pos_a = camera_corners()[0]
    camera_lens_pos_b = camera_corners()[1]
    camera_lens_pos_c = camera_corners()[2]

    # get the arrays for looping through
    xy_array = create_screen_array_xy(camera_lens_pos_a, camera_lens_pos_b)
    z_array = create_screen_array_z(camera_lens_pos_a, camera_lens_pos_c)

    compute_data = []

    # loop through every pixel and call the tracing function on multiple cores
    for i, xy in enumerate(xy_array):
        for j, z in enumerate(z_array):
            compute_data.append([i, xy, j, z])

    compute_results = []

    # simultaneously compute the color of each pixel on the threads created with Pool()
    compute_results = pool.map_async(start_tracing, compute_data).get()
    # compute_results = pool.map(start_tracing, compute_data)

    # use the results to fill the image pixel by pixel
    for pixel in list(compute_results):
        i, j, color = pixel

        # gamma correction
        color = color ** gamma

        screen.fill(255*color, (i, j, 1, 1))


class PhotonMap():

    def __init__(self, kdtree, hashmap, hmc, kdtc):
        self.kdtree = kdtree
        self.hashmap = hashmap

        self.hashmap_caustics = hmc
        self.kdtree_caustics = kdtc

    def getKDtree(self):
        return self.kdtree

    def getHashmap(self):
        return self.hashmap

    def getTreeCaustics(self):
        return self.kdtree_caustics

    def getHashmapCaustics(self):
        return self.hashmap_caustics


def init_thread(args):
    global photon_map_mp
    photon_map_mp = args

if __name__ == '__main__':  # needed for multiprocessing
    # p = shoot_photons()
    # kdtc = shoot_caustics()
    # photon_map_mp = PhotonMap(p, photon_map_dict, caustics_dict, kdtc)

    mp.freeze_support()  # fixes some bug with windows, not necessary on linux TT.TT
    # create four threads on two cores for parallel computing
    # pool = mp.Pool(processes=4, initializer=init_thread, initargs=(photon_map_mp,))
    pool = mp.Pool(processes=4)

    # initiate python and window
    pygame.init()
    screen_w, screen_h = 400, 300
    screen = pygame.display.set_mode((screen_w, screen_h))

    clock = pygame.time.Clock()
    clock.tick()

    screen.fill((0, 0, 0))  # fill screen with black
    draw_scene()  # draw each pixel with the ray tracing method
    print('done', clock.tick() / 1000, 'seconds')
    pygame.image.save(screen, 'traced_img.png')
    # game loop
    while True:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # exit the program
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()  # switch frame buffers