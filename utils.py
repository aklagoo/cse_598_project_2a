import os
import os.path
from collections import namedtuple
import numpy as np
from math import radians, tan
import cv2
import const


def write_arrays(path, arrays):
    """Writes arrays to an XML file using cv2.FileStorage

    Args:
        path: Path to the XML file
        arrays: Dictionary containing arrays indexed by keys
    """
    writer = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for key in arrays:
        writer.write(key, arrays[key])
    writer.release()


def read_arrays(path, keys, tuple_id: str = 'Parameters'):
    """Reads a namedtuple of arrays from an XML file using cv2.FileStorage.

    Args:
        path: Path to the XML file
        keys: Keys to access the arrays
        tuple_id: Tuple typename
    """
    reader = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    Arrays = namedtuple(tuple_id, keys)
    _arrays = {}

    # TODO Raise exception for unavailable keys
    for key in keys:
        _arrays[key] = reader.getNode(key).mat()
    reader.release()

    return Arrays(**_arrays)


def load_intrinsics(dir_params: str = const.DIR_PARAMS):
    """Loads intrinsics for left and right cameras from files.

    Args:
        dir_params: Path to the intrinsics/parameters directory

    Returns:
        Namedtuple containing:
        `left camera matrix, left distortion coefficients, right camera matrix,
        right distortion coefficients`
    """
    Intrinsics = namedtuple('Intrinsics', 'cml cmr dsl dsr')

    # Read left camera intrinsics
    reader = cv2.FileStorage(os.path.join(dir_params, 'left_camera_intrinsics.xml'), cv2.FILE_STORAGE_READ)
    cml = reader.getNode('cml').mat()
    dsl = reader.getNode('dsl').mat()
    reader.release()

    # Read left camera intrinsics
    reader = cv2.FileStorage(os.path.join(dir_params, 'right_camera_intrinsics.xml'), cv2.FILE_STORAGE_READ)
    cmr = reader.getNode('cmr').mat()
    dsr = reader.getNode('dsr').mat()
    reader.release()

    return Intrinsics(cml, cmr, dsl, dsr)


def load_st_params(dir_params: str = const.DIR_PARAMS):
    """Load stereo calibration and rectification matrices.

    Args:
        dir_params: Path to the calibration and rectification parameters directory

    Returns:
        Two namedtuples containing stereo calibration and rectification parameters
    """
    StereoCalibParams = namedtuple('StereoCalibParams', 'R T E F')
    StereoRectParams = namedtuple('StereoRectParams', 'Rl Rr Pl Pr Q')

    # Read stereo calibration parameters
    reader = cv2.FileStorage(os.path.join(dir_params, 'stereo_calibration.xml'), cv2.FILE_STORAGE_READ)
    R = reader.getNode('R').mat()
    T = reader.getNode('T').mat()
    E = reader.getNode('E').mat()
    F = reader.getNode('F').mat()
    reader.release()

    # Read stereo rectification parameters
    reader = cv2.FileStorage(os.path.join(dir_params, 'stereo_rectification.xml'), cv2.FILE_STORAGE_READ)
    Rl = reader.getNode('Rl').mat()
    Rr = reader.getNode('Rr').mat()
    Pl = reader.getNode('Pl').mat()
    Pr = reader.getNode('Pr').mat()
    Q = reader.getNode('Q').mat()
    reader.release()

    return StereoCalibParams(R, T, E, F), StereoRectParams(Rl, Rr, Pl, Pr, Q)


def plot_camera(ax, r=np.identity(3), t=np.zeros((3, 1)), theta_x=45, theta_y=45, f=1):
    """Draws camera in 3D space.

    Args:
        ax: A 3D matplotlib subplot
        r: A rotation matrix. Defaults to an identity matrix for no rotation
        t: A translation matrix. Defaults to zero for no translation
        theta_x: Horizontal field of view
        theta_y: Vertical field of view
        f: Focal length of camera
    """
    # Calculate tan(x) and tan(y)
    tan_x = tan(radians(theta_x))
    tan_y = tan(radians(theta_y))

    # Camera polygon vertices
    vertices = np.asarray([
        (0, 0, 0),
        (-tan_x, -tan_y, 1),
        (0, 0, 0),
        (-tan_x, tan_y, 1),
        (0, 0, 0),
        (tan_x, -tan_y, 1),
        (0, 0, 0),
        (tan_x, tan_y, 1),
        (tan_x, -tan_y, 1),
        (-tan_x, -tan_y, 1),
        (-tan_x, tan_y, 1),
        (tan_x, tan_y, 1)
    ]) * f
    # Rotate and translate camera
    vertices = np.dot(vertices, r).T + t

    # Draw camera polygon
    ax.plot(*vertices, color='black')


def load_img_pair(task_num, dir_img: str = const.DIR_IMG, indices: tuple = (0, 0), gray=False):
    """Loads chessboard images from left and right cameras.

    Args:
        task_num: Task ID
        dir_img: Path to the image directory
        indices: Indices to select left and right
        gray: Boolean indicating whether the image is to be loaded as a grayscale image or not

    Returns:
        Two NumPy arrays containing image data
    """
    if gray:
        color = cv2.IMREAD_GRAYSCALE
    else:
        color = cv2.IMREAD_COLOR
    d = ['1', '2', '3_and_4', '3_and_4']
    img_l = cv2.imread(os.path.join(dir_img, 'task_{0}\\left_{1}.png'.format(d[task_num - 1], indices[0])), color)
    img_r = cv2.imread(os.path.join(dir_img, 'task_{0}\\right_{1}.png'.format(d[task_num - 1], indices[1])), color)

    return img_l, img_r


def extract_crn_2d_3d(img_l, img_r, grid_size=const.SIZE_GRID):
    """Detects chessboard corners and creates 3D object points.

    Args:
        img_l: Image from left camera
        img_r: Image from right camera
        grid_size: Tuple of size (rows, widths)

    Returns:
        Detected corners crn_l and crn_r
        Corresponding object points obj_pt
    """
    # Find corners from left and right images
    crn_l = cv2.findChessboardCorners(img_l, const.SIZE_GRID)[1]
    crn_r = cv2.findChessboardCorners(img_r, const.SIZE_GRID)[1]

    # Generate 3D object points corresponding to chessboard corners
    obj_pt = np.concatenate((
        np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2),
        np.zeros((grid_size[0] * grid_size[1], 1))
    ), axis=1).astype('float32')

    return crn_l, crn_r, obj_pt


def load_img_batch(task_num, img_dir=const.DIR_IMG, gray=False):
    """Extracts left and right images from directory. This function is a batch version of utils.load_img_pair.

    Args:
        task_num: Task ID
        img_dir: Path to the image directory
        gray: Boolean indicating whether the images are to be loaded as grayscale images or not

    Returns:
        Two arrays containing images from the left and right cameras respectively
    """
    # Get list of files
    d = ['1', '2', '3_and_4', '3_and_4']
    base_path = os.path.join(img_dir, 'task_{0}'.format(d[task_num - 1]))
    files = [(os.path.join(base_path, x), x) for x in os.listdir(base_path)]

    # Sort files and load images
    images_l, images_r = [], []
    for file in files:
        if file[1][0] == 'l':
            images_l.append(cv2.imread(file[0]))
        else:
            images_r.append(cv2.imread(file[0]))

    return images_l, images_r
