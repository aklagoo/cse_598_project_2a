import numpy as np
import math


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
    tan_x = math.tan(math.radians(theta_x))
    tan_y = math.tan(math.radians(theta_y))

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
