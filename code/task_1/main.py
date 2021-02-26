import const
import os.path
import cv2
import numpy as np


def get_images(img_dir):
    """Extracts images from directory"""
    # Get list of files
    base_path = os.path.join(img_dir, 'task_1')
    files = [(os.path.join(base_path, x), x) for x in os.listdir(base_path)]
    
    # Sort files and load images
    img_l = []
    img_r = []
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

    _, _cam_mtx_l, _dst_l, _, _ = cv2.calibrateCamera(obj_pt, crn_l, const.SIZE_IMG, None, None)
    _, _cam_mtx_r, _dst_r, _, _ = cv2.calibrateCamera(obj_pt, crn_r, const.SIZE_IMG, None, None)
    
    return _cam_mtx_l, _dst_l, _cam_mtx_r, _dst_r


if __name__ == '__main__':
    # Calibrate camera
    cam_mtx_l, dst_l, cam_mtx_r, dst_r = calibrate_camera()

    # Extract and distort images
    im2_l = cv2.imread(os.path.join(const.DIR_IMG, 'task_1\\left_2.png'))
    im2_r = cv2.imread(os.path.join(const.DIR_IMG, 'task_1\\right_2.png'))
    im2_l = cv2.undistort(im2_l, cam_mtx_l, dst_l, None)
    im2_r = cv2.undistort(im2_r, cam_mtx_r, dst_r, None)

    # Write files
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\left_2.png'), im2_l)
    cv2.imwrite(os.path.join(const.DIR_OUT, 'task_1\\right_2.png'), im2_r)
    np.savetxt(os.path.join(const.DIR_PARAMS, 'left_camera_intrinsics_cam_mtx.csv'), cam_mtx_l)
    np.savetxt(os.path.join(const.DIR_PARAMS, 'left_camera_intrinsics_dst.csv'), dst_l)
    np.savetxt(os.path.join(const.DIR_PARAMS, 'right_camera_intrinsics_cam_mtx.csv'), cam_mtx_r)
    np.savetxt(os.path.join(const.DIR_PARAMS, 'right_camera_intrinsics_dst.csv'), dst_r)
