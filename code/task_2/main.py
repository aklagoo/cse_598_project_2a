import utils
import const
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_img(dir_img: str = const.DIR_IMG, indices: tuple = (0, 0)):
    """Loads chessboard images from left and right cameras.

    Args:
        dir_img: Path to the image directory
        indices: Indices to select left and right images

    Returns:
        Two NumPy arrays containing image data
    """
    img_l = cv2.imread(os.path.join(dir_img, 'task_2\\left_{0}.png'.format(indices[0])))
    img_r = cv2.imread(os.path.join(dir_img, 'task_2\\right_{0}.png'.format(indices[1])))

    return img_l, img_r


def load_intrinsics(dir_params: str = const.DIR_PARAMS):
    """Loads intrinsics for left and right cameras from files.

    Args:
        dir_params: Path to the intrinsics/parameters directory

    Returns:
        Four NumPy arrays containing camera intrinsic matrices and distortion
        coefficients for each camera in the following order:

        `left camera matrix, left distortion coefficients, right camera matrix,
        right distortion coefficients`
    """
    left_camera_intrinsics = utils.read_arrays(
        os.path.join(const.DIR_PARAMS, 'left_camera_intrinsics.xml'),
        ['cam_mtx_l', 'dst_l'])
    right_camera_intrinsics = utils.read_arrays(
        os.path.join(const.DIR_PARAMS, 'right_camera_intrinsics.xml'),
        ['cam_mtx_r', 'dst_r'])

    return left_camera_intrinsics['cam_mtx_l'], left_camera_intrinsics['dst_l'], \
           right_camera_intrinsics['cam_mtx_r'], right_camera_intrinsics['dst_r']


def stereo_calibrate(img, intrinsics):
    """Performs camera calibration and returns rotation, translation, fundamental and essential matrices"""
    # Extract data from arguments
    img_l, img_r = img
    cam_mtx_l, dst_l, cam_mtx_r, dst_r = intrinsics

    # Find corners from left and right images
    crn_l = cv2.findChessboardCorners(img_l, const.SIZE_GRID)[1]
    crn_r = cv2.findChessboardCorners(img_r, const.SIZE_GRID)[1]

    # Generate 3D object points corresponding to chessboard corners
    obj_pt = np.concatenate((
        np.mgrid[0:const.SIZE_GRID[0], 0:const.SIZE_GRID[1]].T.reshape(-1, 2),
        np.zeros((const.SIZE_GRID[0] * const.SIZE_GRID[1], 1))
    ), axis=1).astype('float32')

    # Perform stereo calibration and extract rotation, translation,
    # essential and fundamental matrices
    _params = cv2.stereoCalibrate(
        np.array([obj_pt]), np.array([crn_l]), np.array([crn_r]),
        cam_mtx_l, dst_l, cam_mtx_r, dst_r, (480, 640),
        flags=cv2.CALIB_FIX_INTRINSIC)
    _, cam_mtx_l, dst_l, cam_mtx_r, dst_r, R, T, E, F = _params

    # In the following section, we verify the calibration by
    # triangulating and plotting these points in 3D. We begin by
    # undistorting the extracted corners.
    #
    # To triangulate these points, we generate projection matrices [I|0]
    # and [R|t]. We then pass these matrices and undistorted points to
    # cv2.triangulatePoints to get homogeneous points. These points are
    # converted to cartesian points and plotted on a graph with a camera.
    undist_crn_l = cv2.undistortPoints(crn_l, cam_mtx_l, dst_l)
    undist_crn_r = cv2.undistortPoints(crn_r, cam_mtx_r, dst_r)
    projection_l = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1).astype('float32')
    projection_r = np.concatenate((R, T), axis=1).astype('float32')

    # Triangulate points and convert to cartesian points
    points = cv2.triangulatePoints(projection_l, projection_r, undist_crn_l, undist_crn_r)
    cartesian_points = points[:3] / points[3]

    # Create subplot and plot cameras and points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*cartesian_points)
    utils.plot_camera(ax)
    utils.plot_camera(ax, R, T)
    plt.show()


if __name__ == '__main__':
    images = load_img()
    parameters = load_intrinsics()

    stereo_calibrate(images, parameters)
