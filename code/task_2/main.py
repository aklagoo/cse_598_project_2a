import utils
import const
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils import extract_crn_2d_3d


def st_calib_rect(img_l, img_r, intr):
    """Performs camera calibration and returns rotation, translation, fundamental and essential matrices

    Args:
        img_l: Image from the left camera
        img_r: Image from the right camera
        intr: Camera intrinsics from the left and right cameras as a namedtuple
    """
    # TODO Refactor function
    # Extract data from arguments
    img_size = img_l.shape[:2]

    # Detect chessboard corners and extract 2D-3D correspondences and convert to NumPy arrays
    crn_l, crn_r, obj_pt = extract_crn_2d_3d(img_l, img_r)
    crn_l = np.asarray([crn_l])
    crn_r = np.asarray([crn_r])
    obj_pt = np.asarray([obj_pt])

    # Perform stereo calibration and extract rotation, translation,
    # essential and fundamental matrices
    _params = cv2.stereoCalibrate(obj_pt, crn_l, crn_r, intr.cml, intr.dsl, intr.cmr, intr.dsr, img_size,
                                  flags=cv2.CALIB_FIX_INTRINSIC)
    _, _, _, _, _, rot, T, E, F = _params

    # Extract stereo rectification parameters
    R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(intr.cml, intr.dsl, intr.cmr, intr.dsr, img_size, rot, T)

    return T, rot, F, E, R_l, R_r, P_l, P_r, Q


def generate_output(crn_l, crn_r, cam_mtx_l, dst_l, cam_mtx_r, dst_r, R, T):
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
    image_left, image_right = utils.load_img_pair(2)
    intrinsics = utils.load_intrinsics()

    T, R, F, E, Rl, Rr, Pl, Pr, Q = st_calib_rect(image_left, image_right, intrinsics)

    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'stereo_calibration.xml'),
                       {
                           'T': T,
                           'R': R,
                           'F': F,
                           'E': E,
                       })
    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'stereo_rectification.xml'),
                       {
                           'Rl': Rl,
                           'Rr': Rr,
                           'Pl': Pl,
                           'Pr': Pr,
                           'Q': Q
                       })
