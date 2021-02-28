import os
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
    """Load stereo calibration and rectification matrices

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


def load_img_pair(task_num, dir_img: str = const.DIR_IMG, indices: tuple = (0, 0)):
    """Loads chessboard images from left and right cameras.

    Args:
        task_num: Task ID
        dir_img: Path to the image directory
        indices: Indices to select left and right images

    Returns:
        Two NumPy arrays containing image data
    """
    d = ['1', '2', '3_and_4', '3_and_4']
    img_l = cv2.imread(os.path.join(dir_img, 'task_{0}\\left_{1}.png'.format(d[task_num - 1], indices[0])))
    img_r = cv2.imread(os.path.join(dir_img, 'task_{0}\\right_{1}.png'.format(d[task_num - 1], indices[1])))

    return img_l, img_r
