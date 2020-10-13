import math
import numpy as np

def polar(x, y):
    """
    returns polar angle theta in degrees
    """

    # r = (x ** 2 + y ** 2) ** .5
    theta = math.degrees(math.atan2(y,x))
    return theta

# thanks to https://stackoverflow.com/questions/52990094/calculate-circle-given-3-points-code-explanation
def circleRadius(b, c, d):
    
    circle = True
    temp = c[0]**2 + c[1]**2
    bc = (b[0]**2 + b[1]**2 - temp) / 2
    cd = (temp - d[0]**2 - d[1]**2) / 2

    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        circle = False
        return 0, 0, circle

    # Center of circle
    cx = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    return cx, cy, circle

def distance(tup1, tup2):
    return math.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

def get_gradient(tagent_tail, tagent_tip):
    dx = tagent_tip[0] - tagent_tail[0] 
    dy = tagent_tip[1] - tagent_tail[1] 
    return dy/dx

def get_angle(tagent_tail, tagent_tip):
    dx = tagent_tip[0] - tagent_tail[0] 
    dy = tagent_tip[1] - tagent_tail[1] 
    return polar(dx, dy)


def get_tagent_bearing(params):
    if params['is_reversed']:
        full_waypoints_list  = params['waypoints'][::-1]
    else:
        full_waypoints_list  = params['waypoints']
    full_waypoints_list.pop()
    car_poistion = (params['x'], params['y'])
    distances = [distance(point, car_poistion) for point in full_waypoints_list]
    
    # get closest waypoints
    min_dist = min(distances)
    closest_index = distances.index(min_dist)

    # get the point before and after
    length = len(full_waypoints_list)
    if closest_index == 0:
        point_before_i = -1
    else: 
        point_before_i = closest_index - 1
    
    if closest_index == length - 1:
        point_after_i = 0
    else:
        point_after_i = closest_index + 1
    pt_before = full_waypoints_list[point_before_i]
    pt_closet = full_waypoints_list[closest_index]
    pt_after = full_waypoints_list[point_after_i]
    cx, cy, circle = circleRadius(pt_before, pt_closet, pt_after)
    if circle:
        radius_gradient = get_gradient((cx, cy), car_poistion)
        tagent_gradient = -1/radius_gradient
        tagent_bearing = math.degrees(math.atan(tagent_gradient))
    else:
        tagent_bearing = get_angle(pt_closet, pt_after)
    return tagent_bearing

def get_ideal_heading(params):
    ideal_heading = get_tagent_bearing(params)
    quotient = math.floor(ideal_heading/360.0)
    remainder = ideal_heading - quotient * 360.0
    if remainder > 180:
        ideal_heading = remainder - 360.0
    else: 
        ideal_heading = remainder

    return ideal_heading


def reward_ideal_heading(params):
    heading = params['heading']
    ideal_heading = get_ideal_heading(params)
    error = heading - ideal_heading
    scores = 1.0 - abs(error/ 360.0) 
    return max(scores, 0.01)

 # sample reward function for centreline
def reward_centre_line(params, reward):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''
    
    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the agent is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward *= 1
    elif distance_from_center <= marker_2:
        reward *= 0.5
    elif distance_from_center <= marker_3:
        reward *= 0.1
    else:
        reward *= 1e-3  # likely crashed/ close to off track
    return float(reward)

def penalise_drifting(params, reward):
    speed = abs(params['speed'])
    steering = abs(params['steering_angle'])
    if speed > 2.5 - (0.4 * abs(steering)):
        reward *= 0.8
    return reward

def straight_line_reward(params, reward):
    speed = abs(params['speed'])
    steering = abs(params['steering_angle'])
    if steering < 0.1 and speed > 3:
        reward *= 1.2
    return reward

def reward_function(params):
    reward = reward_ideal_heading(params)
    reward = reward_centre_line(params, reward)
    reward = penalise_drifting(params, reward)
    reward = straight_line_reward(params, reward)
    return reward
