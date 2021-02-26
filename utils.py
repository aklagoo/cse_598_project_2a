import numpy as np


def plot_camera(ax, r=np.identity(3), t=np.zeros((3, 1))):
    # Constants assumed
    tan_x, tan_y, f = 1, 1, 1

    points = np.asarray([
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

    points = np.dot(points, r).T + t

    ax.plot(*points, color='black')
