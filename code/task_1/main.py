import const
import utils
import os.path
import cv2
import numpy as np


def load_img_batch(img_dir=const.DIR_IMG):
    """Extracts left and right images from directory. This function is a batch version of utils.load_img_pair.

    Args:
        img_dir: Path to the image directory

    Returns:
        Two arrays containing images from the left and right cameras respectively
    """
    # Get list of files
    base_path = os.path.join(img_dir, 'task_1')
    files = [(os.path.join(base_path, x), x) for x in os.listdir(base_path)]
    
    # Sort files and load images
    images_l, images_r = [], []
    for file in files:
        if file[1][0] == 'l':
            images_l.append(cv2.imread(file[0]))
        else:
            images_r.append(cv2.imread(file[0]))

    return images_l, images_r


def extract_crn_2d_3d_batch(images_l, images_r, size_grid=const.SIZE_GRID):
    """Detects chessboard corners and creates 3D object points.

    Args:
        images_l: Image from left camera
        images_r: Image from right camera
        size_grid: Tuple of size (rows, widths)

    Returns:
        Detected corners crn_l and crn_r
        Corresponding object points obj_pt
    """
    # Get batch information
    batch_size = len(images_l)

    # Find corners from left and right images
    corners_l = np.array([cv2.findChessboardCorners(img, size_grid)[1] for img in images_l])
    corners_r = np.array([cv2.findChessboardCorners(img, size_grid)[1] for img in images_r])

    # Generate 3D object points corresponding to chessboard corners
    obj_points = np.concatenate((
        np.mgrid[0:size_grid[0], 0:size_grid[1]].T.reshape(-1, 2),
        np.zeros((size_grid[0] * size_grid[1], 1))
    ), axis=1).astype('float32')
    obj_points = np.array([obj_points] * batch_size, dtype=np.float32)

    return corners_l, corners_r, obj_points


def calibrate_camera(corners_l, corners_r, obj_points, size_image):
    """Calibrate left and right camera using images.

    Args:
        corners_l: Corners extracted from the left image
        corners_r: Corners extracted from the right image
        obj_points: 3D object points corresponding to the corners
        size_image: Image dimensions as a tuple

    Returns:
        Two parameters for camera intrinsics and distortion coefficients for each camera
    """
    _, _cml, _dsl, _, _ = cv2.calibrateCamera(obj_points, corners_l, size_image, None, None)
    _, _cmr, _dsr, _, _ = cv2.calibrateCamera(obj_points, corners_r, size_image, None, None)
    
    return _cml, _dsl, _cmr, _dsr


def write_output(image_l, image_r, corners_l, corners_r, cam_intrinsic_l, distortion_l, cam_intrinsic_r, distortion_r):
    """Undistorts images and writes images to files.

    Args:
        image_l: Image from the left camera
        image_r: Image from the right camera
        corners_l: Chessboard corners detected from the left camera
        corners_r: Chessboard corners detected for the right camera
        cam_intrinsic_l: Camera intrinsic matrix for the left camera
        distortion_l: Distortion coefficients of the left camera
        cam_intrinsic_r: Camera intrinsic matrix for the right camera
        distortion_r: Distortion coefficients of the right camera
    """
    # Extract and undistort images
    im2_l_backup, im2_r_backup = np.copy(image_l), np.copy(image_r)
    im2_l_chk = cv2.drawChessboardCorners(image_l, const.SIZE_GRID, corners_l[3], True)
    im2_r_chk = cv2.drawChessboardCorners(image_r, const.SIZE_GRID, corners_r[3], True)
    im2_l_undist = cv2.undistort(im2_l_backup, cam_intrinsic_l, distortion_l, None)
    im2_r_undist = cv2.undistort(im2_r_backup, cam_intrinsic_r, distortion_r, None)

    # Write files
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\left_2.png'), image_l)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\right_2.png'), image_r)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\left_2_chk.png'), im2_l_chk)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\right_2_chk.png'), im2_r_chk)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\left_2_undist.png'), im2_l_undist)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\right_2_undist.png'), im2_r_undist)


def main():
    # Load images
    img_l, img_r = load_img_batch()
    img_shape = img_l[0].shape[:2]

    # Extract point correspondences and calibrate the camera
    crn_l, crn_r, obj_pt = extract_crn_2d_3d_batch(img_l, img_r)
    cml, dsl, cmr, dsr = calibrate_camera(crn_l, crn_r, obj_pt, img_shape)

    # Write outputs
    IMG_ID = 2
    im2_l, im2_r = utils.load_img_pair(1, indices=(IMG_ID, IMG_ID))
    write_output(im2_l, im2_r, crn_l[IMG_ID + 1], crn_r[IMG_ID + 1], cml, dsl, cmr, dsr)

    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'left_camera_intrinsics.xml'),
                       {'cml': cml, 'dsl': dsl})
    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'right_camera_intrinsics.xml'),
                       {'cmr': cml, 'dsr': dsr})


if __name__ == '__main__':
    main()
