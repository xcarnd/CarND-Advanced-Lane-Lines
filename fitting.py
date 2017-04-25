import numpy as np
import matplotlib.pyplot as plt


def sample(polyline_params, y_range, abs_x_noise, ratio=1.0):
    """Sample from the specified polyline.
    """
    a, b, c = polyline_params
    y = np.linspace(y_range[0], y_range[1] - 1, y_range[1] - y_range[0])
    x = a * y ** 2 + b * y + c
    x = x + np.array([np.random.randint(-abs_x_noise, high=abs_x_noise + 1) for i in range(len(y))])
    pts = np.stack((y, x), axis=1)
    np.random.shuffle(pts)
    return pts[:int(len(y) * ratio)]


def average_polylines(polylines, weights, y_range, abs_x_noise):
    """Average the given polylines using the given weights.
    """
    assert len(polylines) == len(weights), "polylines and weights must be of the same length"
    samples = []
    for i in range(len(polylines)):
        polyline = polylines[i]
        ratio = weights[i]
        pts = sample(polyline, y_range, abs_x_noise, ratio)
        samples.append(pts)
    pts = np.concatenate(samples)
    avg = np.polyfit(pts[:, 0], pts[:, 1], 2)
    return avg
