import math
import numpy as np

 
def reward_centre_line(params):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''
    
    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle

    # Calculate 3 markers that are at varying distances away from the center line
    if distance_from_center > 0.5*track_width:
        reward = 0  # likely crashed/ close to off track
    else:
        reward = (1 - 1*distance_from_center/(0.5*track_width))**(1/16)
    return float(reward)



def reward_function(params):
    reward = reward_centre_line(params)
  
    return reward