import const
import utils
import os.path
import cv2
import numpy as np


def get_images(img_dir):
    """Extracts left and right images from directory"""
    # Get list of files
    base_path = os.path.join(img_dir, 'task_1')
    files = [(os.path.join(base_path, x), x) for x in os.listdir(base_path)]
    
    # Sort files and load images
    img_l, img_r = [], []
    for file in files:
        if file[1][0] == 'l':
            img_l.append(cv2.imread(file[0]))
        else:
            img_r.append(cv2.imread(file[0]))
    
    return len(img_l), img_l, img_r


def calibrate_camera(img_dir=const.DIR_IMG):
    """Calibrate left and right camera using images"""
    # Read images
    num_img, img_l, img_r = get_images(img_dir)
    
    # Get corners for left and right cameras
    crn_l = np.array([cv2.findChessboardCorners(img, const.SIZE_GRID)[1] for img in img_l])
    crn_r = np.array([cv2.findChessboardCorners(img, const.SIZE_GRID)[1] for img in img_r])
    
    # Create object points
    obj_pt = np.concatenate((
        np.mgrid[0:const.SIZE_GRID[0], 0:const.SIZE_GRID[1]].T.reshape(-1, 2),
        np.zeros((const.SIZE_GRID[0] * const.SIZE_GRID[1], 1))
    ), axis=1)
    obj_pt = np.array([obj_pt]*num_img, dtype=np.float32)

    _, _cml, _dsl, _, _ = cv2.calibrateCamera(obj_pt, crn_l, const.SIZE_IMG, None, None)
    _, _cmr, _dsr, _, _ = cv2.calibrateCamera(obj_pt, crn_r, const.SIZE_IMG, None, None)
    
    return _cml, _dsl, _cmr, _dsr


if __name__ == '__main__':
    # Calibrate camera
    cml, dsl, cmr, dsr = calibrate_camera()

    # Extract and distort images
    im2_l, im2_r = utils.load_img_pair(1, indices=(2,2))
    im2_l = cv2.undistort(im2_l, cml, dsl, None)
    im2_r = cv2.undistort(im2_r, cmr, dsr, None)

    # Write files
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\left_2.png'), im2_l)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\right_2.png'), im2_r)

    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'left_camera_intrinsics.xml'),
                       {'cml': cml, 'dsl': dsl})
    utils.write_arrays(os.path.join(const.DIR_PARAMS, 'right_camera_intrinsics.xml'),
                       {'cmr': cml, 'dsr': dsr})
