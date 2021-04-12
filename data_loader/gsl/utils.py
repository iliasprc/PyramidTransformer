import glob
import json

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

NUM_POSE_JOINTS = 33 - 10
NUM_HAND_JOINTS = 21
NUM_FACE_JOINTS = 500
MISSING = -1000.0


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def read_json_sequence(path='/**.json', do_smoothing=True):
    pose_jsons = sorted(glob.glob(
        path + '/**.json'))

    RH = 0
    LH = 0
    pose = []
    lh = []
    rh = []
    for i in pose_jsons:
        pose_joints, left_hand_joints, right_hand_joints, lh_missing, rh_missing = read_blazepose(i)
        RH += rh_missing
        LH += lh_missing
        pose.append(pose_joints)
        lh.append(left_hand_joints)
        rh.append(right_hand_joints)
    #print(len(pose),path)
    pose = np.stack(pose)
    lh = np.stack(lh)
    rh = np.stack(rh)
    # print(pose)
    if do_smoothing:
        pose = smooth(pose)
        # print(pose)
        lh = smooth(lh)
        rh = smooth(rh)
    return pose, lh, rh


def read_blazepose(json_path):
    with open(
            json_path) as f:
        data = json.load(f)
        #print(data.keys())
        pose = data['pose']
        left_hand = data['left_hand']
        right_hand = data['right_hand']
        face = data['face']
        face_landmarks = data['face_landmarks']

        # print(len(pose))
        pose_joints = []
        if pose==[]:
            for i in range(NUM_POSE_JOINTS):
                joint = [MISSING, MISSING, MISSING, MISSING]
                pose_joints.append(joint)
            pose_joints = np.stack(pose_joints)
        else:
            for i in range(NUM_POSE_JOINTS):
                # print(pose[i][f'pose_{str(i)}'])
                joint = [pose[i][f'pose_{str(i)}']['X'], pose[i][f'pose_{str(i)}']['Y'], pose[i][f'pose_{str(i)}']['Z'],
                         pose[i][f'pose_{str(i)}']['Visibility']]
                pose_joints.append(joint)
            pose_joints = np.stack(pose_joints)
        # print(pose_joints.shape)

        left_hand_joints = []
        # print(left_hand)
        if left_hand == []:
            lh_missing = 1
            for i in range(NUM_HAND_JOINTS):
                joint = [MISSING, MISSING, MISSING, MISSING]

                left_hand_joints.append(joint)
            left_hand_joints = np.stack(left_hand_joints)
        else:
            lh_missing = 0
            for i in range(NUM_HAND_JOINTS):
                # print(left_hand[i][f'pose_{str(i)}'], i)

                joint = [left_hand[i][f'left_hand_{str(i)}']['X'], left_hand[i][f'left_hand_{str(i)}']['Y'],
                         left_hand[i][f'left_hand_{str(i)}']['Z'],
                         left_hand[i][f'left_hand_{str(i)}']['Visibility']]
                left_hand_joints.append(joint)
            left_hand_joints = np.stack(left_hand_joints)
        # print(left_hand_joints.shape)
        # print(right_hand)
        right_hand_joints = []
        if right_hand == []:
            rh_missing = 1
            for i in range(NUM_HAND_JOINTS):
                joint = [MISSING, MISSING, MISSING, MISSING]

                right_hand_joints.append(joint)
            right_hand_joints = np.stack(right_hand_joints)
        else:
            rh_missing = 0
            for i in range(NUM_HAND_JOINTS):
                # print(right_hand[i][f'right_hand_{str(i)}'], i)

                joint = [right_hand[i][f'right_hand_{str(i)}']['X'], right_hand[i][f'right_hand_{str(i)}']['Y'],
                         right_hand[i][f'right_hand_{str(i)}']['Z'],
                         right_hand[i][f'right_hand_{str(i)}']['Visibility']]
                right_hand_joints.append(joint)
            right_hand_joints = np.stack(right_hand_joints)
        # print(right_hand_joints.shape)

    return pose_joints, left_hand_joints, right_hand_joints, lh_missing, rh_missing


def smooth(array):
    assert len(array.shape) == 3
    time = array.shape[0]
    if time > 21:
        wnds = [21, 11]
        orders = [3, 4]
    else:
        t = int(round_up_to_odd(time))
        wnds = [t - 2, t - 4]
        orders = [1, 2]
        # print(wnds)
    for i in range(array.shape[1]):
        # print('JOINT ', i)
        # print(array.shape)
        array[:, i, 0] = savitzky_golay_filtering(array[:, i, 0], wnds=wnds, orders=orders)
        array[:, i, 1] = savitzky_golay_filtering(array[:, i, 1], wnds=wnds, orders=orders)
        array[:, i, 2] = savitzky_golay_filtering(array[:, i, 2], wnds=wnds, orders=orders)
    return array


def savitzky_golay_filtering(timeseries, wnds=[21, 11], orders=[3, 4], maxNanElems=250):
    # print(wnds)
    timeseries[timeseries == MISSING] = np.nan
    if np.sum(np.isnan(timeseries)) == timeseries.shape[0]:
        # print('GAMW',timeseries.shape)
        timeseries[0] = 0.0
        timeseries[-1] = 0.1
    interp_ts = pd.Series(timeseries)
    interp_ts = interp_ts.interpolate(method='linear', limit=maxNanElems, limit_direction='both')
    smooth_ts = interp_ts
    assert (np.sum(np.isnan(smooth_ts)) == 0), print(np.sum(np.isnan(smooth_ts)), wnds, timeseries)
    wnd, order = wnds[0], orders[0]
    F = 1e8
    W = None
    it = 0
    counter = 0
    while counter < 5:
        smoother_ts = savgol_filter(smooth_ts, window_length=wnd, polyorder=order)
        diff = smoother_ts - interp_ts
        sign = diff > 0
        if W is None:
            W = 1 - np.abs(diff) / np.max(np.abs(diff)) * sign
            wnd, order = wnds[1], orders[1]
        fitting_score = np.sum(np.abs(diff) * W)
        # print it, ' : ', fitting_score
        # print(fitting_score)
        if fitting_score > F:
            break
        else:
            F = fitting_score
            it += 1
        smooth_ts = smoother_ts * sign + interp_ts * (1 - sign)
        counter += 1
    return smooth_ts.to_numpy()
