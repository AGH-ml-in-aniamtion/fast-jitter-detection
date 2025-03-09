import numpy as np


# Credit to Motion In-betweening via Two-stage Transformers
def euler_to_matrix9D(euler, order="zyx", unit="degrees"):
    """
    Euler angle to 3x3 rotation matrix.

    Args:
        euler (ndarray): Euler angle. Shape: (..., 3)
        order (str, optional):
            Euler rotation order AND order of parameter euler.
            E.g. "yxz" means parameter "euler" is (y, x, z) and
            rotation order is z applied first, then x, finally y.
            i.e. p' = YXZp, where p' and p are column vectors.
            Defaults to "zyx".
        unit (str, optional):
            Can be either degrees and radians.
            Defaults to degrees.
    """
    mat = np.identity(3)
    if unit == "degrees":
        euler_radians = euler / 180.0 * np.pi
    elif unit == "radians":
        euler_radians = euler
    else:
        raise RuntimeError("Invalid unit: {}".format(unit))

    for idx, axis in enumerate(order):
        angle_radians = euler_radians[..., idx:idx + 1]

        # shape: (..., 1)
        sin = np.sin(angle_radians)
        cos = np.cos(angle_radians)

        # shape(..., 4)
        rot_mat = np.concatenate([cos, sin, sin, cos], axis=-1)
        # shape(..., 2, 2)
        rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 2, 2)

        if axis == "x":
            rot_mat *= np.array([[1, -1], [1, 1]])
            rot_mat = np.insert(rot_mat, 0, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 0, [1, 0, 0], axis=-1)
        elif axis == "y":
            rot_mat *= np.array([[1, 1], [-1, 1]])
            rot_mat = np.insert(rot_mat, 1, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 1, [0, 1, 0], axis=-1)
        else:
            rot_mat *= np.array([[1, -1], [1, 1]])
            rot_mat = np.insert(rot_mat, 2, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 2, [0, 0, 1], axis=-1)

        mat = np.matmul(mat, rot_mat)

    return mat
