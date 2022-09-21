from ctypes import pointer
#if put in the OpenCV code, need put the target in class to update the target point change
#also put thesteering angle inclas.
k = 0.5  # control gain adjust base on real test

dt = 0.1  # [s] time difference
def stanley_control(state, current_streeingangle):
    """
 return the steering
    """
    current_target_idx, error_front_axle = calc_target_index(state.pointe)#put state in the class

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(current_streeingangle)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
min or max turn 

    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state):
    """
point

    """
    #center_pointx=now camera cantrter
    #center_pointy=now camera cantrter

    # Search nearest point index
    dx = [ state.centerx]
    dy = [ state.centery]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector

    error_front_axle = [ center_pointx-state.centerx,center_pointy-state.centery]

    return target_idx, error_front_axle