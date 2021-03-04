import utils
import const
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import namedtuple


def st_calib_rect(img_l, img_r, intr):
    """Performs camera calibration and returns rotation, translation, fundamental and essential matrices

    Args:
        img_l: Image from the left camera
        img_r: Image from the right camera
        intr: Camera intrinsics from the left and right cameras as a namedtuple

    Returns:
        Two namedtuples containing stereo calibration and rectification parameters
    """
    # Create namedtuples
    StereoCalibParams = namedtuple('StereoCalibParams', 'R T E F')
    StereoRectParams = namedtuple('StereoRectParams', 'Rl Rr Pl Pr Q')

    # Extract data from arguments
    img_size = img_l.shape[:2]

    # Detect chessboard corners and extract 2D-3D correspondences and convert to NumPy arrays
    crn_l, crn_r, obj_pt = utils.extract_crn_2d_3d(img_l, img_r)
    crn_l = np.asarray([crn_l])
    crn_r = np.asarray([crn_r])
    obj_pt = np.asarray([obj_pt])

    # Perform stereo calibration and extract rotation, translation, essential and fundamental matrices
    _params = cv2.stereoCalibrate(obj_pt, crn_l, crn_r, intr.cml, intr.dsl, intr.cmr, intr.dsr, img_size,
                                  flags=cv2.CALIB_FIX_INTRINSIC)
    _, _, _, _, _, R, T, E, F = _params
    st_calib = StereoCalibParams(R, T, E, F)

    # Extract stereo rectification parameters
    Rl, Rr, Pl, Pr, Q, _, _ = cv2.stereoRectify(intr.cml, intr.dsl, intr.cmr, intr.dsr, img_size, R, T)
    st_rect = StereoRectParams(Rl, Rr, Pl, Pr, Q)

    return st_calib, st_rect


def generate_output(image_l, image_r, intrinsics, stereo_calib_params, stereo_rect_params, size_grid):
    # In the following section, we verify the calibration by triangulating and plotting these points in 3D. We begin by
    # undistorting the extracted corners.
    #
    # To triangulate these points, we generate projection matrices [I|0] and [R|t]. We then pass these matrices and
    # undistorted points to cv2.triangulatePoints to get homogeneous points. These points are converted to cartesian
    # points and plotted on a graph with a camera.
    img_size = image_l.shape[:2]
    _, crn_l = cv2.findChessboardCorners(image_l, size_grid)
    _, crn_r = cv2.findChessboardCorners(image_r, size_grid)

    undist_crn_l = cv2.undistortPoints(crn_l, intrinsics.cml, intrinsics.dsl)
    undist_crn_r = cv2.undistortPoints(crn_r, intrinsics.cmr, intrinsics.dsr)
    projection_l = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1).astype('float32')
    projection_r = np.concatenate((stereo_calib_params.R, stereo_calib_params.T), axis=1).astype('float32')

    # Triangulate points and convert to cartesian points
    points = cv2.triangulatePoints(projection_l, projection_r, undist_crn_l, undist_crn_r)
    cartesian_points = points[:3] / points[3]

    # Create subplot and plot cameras and points
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*cartesian_points)
    utils.plot_camera(ax)
    utils.plot_camera(ax, stereo_calib_params.R, stereo_calib_params.T)

    # Undistort and rectify images
    image_l_undist = cv2.undistort(image_l, intrinsics.cml, intrinsics.dsl, None)
    image_r_undist = cv2.undistort(image_r, intrinsics.cmr, intrinsics.dsr, None)

    maps_l = cv2.initUndistortRectifyMap(intrinsics.cml, intrinsics.dsl, stereo_rect_params.Rl, stereo_rect_params.Pl,
                                         img_size, cv2.CV_32FC1)
    maps_r = cv2.initUndistortRectifyMap(intrinsics.cmr, intrinsics.dsr, stereo_rect_params.Rr, stereo_rect_params.Pr,
                                         img_size, cv2.CV_32FC1)
    image_l_rect = cv2.remap(image_l_undist, maps_l[0], maps_l[1], cv2.INTER_LINEAR)
    image_r_rect = cv2.remap(image_r_undist, maps_r[0], maps_r[1], cv2.INTER_LINEAR)

    # Find chessboard corners on each image
    _, crn_undist_l = cv2.findChessboardCorners(image_l_undist, size_grid)
    _, crn_undist_r = cv2.findChessboardCorners(image_r_undist, size_grid)
    _, crn_rect_l = cv2.findChessboardCorners(image_l_rect, size_grid)
    _, crn_rect_r = cv2.findChessboardCorners(image_r_rect, size_grid)

    # Draw chessboard corners
    image_l = cv2.drawChessboardCorners(image_l.copy(), size_grid, crn_l, True)
    image_r = cv2.drawChessboardCorners(image_r.copy(), size_grid, crn_r, True)
    image_l_undist = cv2.drawChessboardCorners(image_l_undist.copy(), size_grid, crn_undist_l, True)
    image_r_undist = cv2.drawChessboardCorners(image_r_undist.copy(), size_grid, crn_undist_r, True)
    image_l_rect = cv2.drawChessboardCorners(image_l_rect.copy(), size_grid, crn_rect_l, True)
    image_r_rect = cv2.drawChessboardCorners(image_r_rect.copy(), size_grid, crn_rect_r, True)

    # Write files
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_l.png'), image_l)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_r.png'), image_r)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_l_undist.png'), image_l_undist)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_r_undist.png'), image_r_undist)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_l_rect.png'), image_l_rect)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'image_r_rect.png'), image_r_rect)

    # Rectified camera
    ax_rect = fig.add_subplot(122, projection='3d')
    utils.plot_camera(ax_rect, stereo_rect_params.Rl)
    utils.plot_camera(ax_rect, stereo_rect_params.Rr, stereo_calib_params.T)

    plt.show()


if __name__ == '__main__':
    image_left, image_right = utils.load_img_pair(2)
    intr = utils.load_intrinsics()

    st_calib, st_rect = st_calib_rect(image_left, image_right, intr)
    generate_output(image_left, image_right, intr, st_calib, st_rect, const.SIZE_GRID)

    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'stereo_calibration.xml'),
                       {
                           'T': st_calib.T,
                           'R': st_calib.R,
                           'F': st_calib.F,
                           'E': st_calib.E,
                       })
    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'stereo_rectification.xml'),
                       {
                           'Rl': st_rect.Rl,
                           'Rr': st_rect.Rr,
                           'Pl': st_rect.Pl,
                           'Pr': st_rect.Pr,
                           'Q': st_rect.Q
                       })
