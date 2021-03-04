import utils
import cv2
import const
import os
from matplotlib import pyplot as plt


def generate_depth_map(img_l, img_r, matcher_params, filter_params):
    """Generates depth map from left and right views.


    """
    # Create left and right StereoSGBM matchers
    left_matcher = cv2.StereoSGBM_create(**matcher_params)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Create WLS filter for map smoothing
    disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    disparity_filter.setLambda(filter_params['lambda'])
    disparity_filter.setSigmaColor(filter_params['sigma'])

    # Generate filtered depth map
    d_l = left_matcher.compute(img_l, img_r).astype('int16')
    d_r = right_matcher.compute(img_r, img_l).astype('int16')
    depth_map = disparity_filter.filter(d_l, img_l, None, d_r)

    return depth_map


def main():
    # Parameters
    matcher_params = {
        'blockSize': 7,
        'minDisparity': 0,
        'numDisparities': 64,
        'preFilterCap': 63,
        'uniquenessRatio': 15,
        'speckleWindowSize': 10,
        'speckleRange': 1,
        'disp12MaxDiff': 20,
        'P1': 8 * 3 * 7 ** 2,
        'P2': 32 * 3 * 7 ** 2,
        'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }

    filter_params = {
        'lambda': 70000,
        'sigma': 1.2
    }

    # Load images and camera parameters
    img_l, img_r = utils.load_img_pair(4, indices=(4, 4), gray=True)
    intr = utils.load_intrinsics()
    calib, rect = utils.load_st_params()

    # Undistort images
    img_l = cv2.undistort(img_l, intr.cml, intr.dsl, rect.Pl)
    img_r = cv2.undistort(img_r, intr.cmr, intr.dsr, rect.Pr)

    # Generate depth map from images
    depth_map = generate_depth_map(img_l, img_r, matcher_params, filter_params)

    # Write to output
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_3\\denoise.png'), depth_map)

    reprojected_image = cv2.reprojectImageTo3D(depth_map, rect.Q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*reprojected_image.T)
    plt.show()
