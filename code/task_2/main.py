import utils
import const
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_img(img_dir=const.DIR_IMG):
    img_l = cv2.imread(os.path.join(img_dir, 'task_2\\left_0.png'))
    img_r = cv2.imread(os.path.join(img_dir, 'task_2\\right_0.png'))

    return img_l, img_r


def load_params(params_dir=const.DIR_PARAMS):
    cam_mtx_l = np.loadtxt(os.path.join(params_dir, 'left_camera_intrinsics_cam_mtx.csv'))
    dst_l = np.loadtxt(os.path.join(params_dir, 'left_camera_intrinsics_dst.csv'))
    cam_mtx_r = np.loadtxt(os.path.join(params_dir, 'right_camera_intrinsics_cam_mtx.csv'))
    dst_r = np.loadtxt(os.path.join(params_dir, 'right_camera_intrinsics_dst.csv'))

    return cam_mtx_l, dst_l, cam_mtx_r, dst_r


def stereo_calibrate(img, params):
    img_l, img_r = img
    cam_mtx_l, dst_l, cam_mtx_r, dst_r = params

    crn_l = cv2.findChessboardCorners(img_l, const.SIZE_GRID)[1]
    crn_r = cv2.findChessboardCorners(img_r, const.SIZE_GRID)[1]

    obj_pt = np.concatenate((
        np.mgrid[0:const.SIZE_GRID[0], 0:const.SIZE_GRID[1]].T.reshape(-1, 2),
        np.zeros((const.SIZE_GRID[0] * const.SIZE_GRID[1], 1))
    ), axis=1).astype('float32')

    params = cv2.stereoCalibrate(np.array([obj_pt]), np.array([crn_l]), np.array([crn_r]), cam_mtx_l, dst_l, cam_mtx_r,
                                 dst_r, (480, 640), flags=cv2.CALIB_FIX_INTRINSIC)
    _, cam_mtx_l, dst_l, cam_mtx_r, dst_r, R, T, E, F = params

    undist_crn_l = cv2.undistortPoints(crn_l, cam_mtx_l, dst_l)
    undist_crn_r = cv2.undistortPoints(crn_r, cam_mtx_r, dst_r)

    projection_l = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1).astype('float32')
    projection_r = np.concatenate((R, T), axis=1).astype('float32')

    points = cv2.triangulatePoints(projection_l, projection_r, undist_crn_l, undist_crn_r)
    cartesian_points = points[:3] / points[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*cartesian_points)
    utils.plot_camera(ax)
    utils.plot_camera(ax, R, T)

    plt.show()

img = load_img()
params = load_params()

stereo_calibrate(img, params)
