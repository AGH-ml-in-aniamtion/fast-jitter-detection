import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def scale_skeleton(positions, rotations):
    pass


def matrix9D_to_euler_angles(mat):
    quat_data = matrix9D_to_quat_torch(mat)
    quat_data = _remove_quat_discontinuities(quat_data)
    rotations = Rotation.from_quat(quat_data.cpu().numpy().flatten().reshape((-1, 4)))
    return rotations.as_euler('ZYX', degrees=True).reshape(*mat.shape[:3], 3)


# Credit to Motion In-betweening via Two-stage Transformers
def matrix9D_to_quat_torch(mat):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        mat : Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{mat.shape}.")
    m00 = mat[..., 0, 0]
    m11 = mat[..., 1, 1]
    m22 = mat[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, mat[..., 2, 1] - mat[..., 1, 2])
    o2 = _copysign(y, mat[..., 0, 2] - mat[..., 2, 0])
    o3 = _copysign(z, mat[..., 1, 0] - mat[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


# Credit to Motion In-betweening via Two-stage Transformers
def fk(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (Tensor): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (Tensor): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D int Tensor): Parent indices.

    Returns:
        Tensor, Tensor: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = torch.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
               torch.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2)


def fk_multi_parent(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (Tensor): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (Tensor): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D int Tensor): Parent indices.

    Returns:
        Tensor, Tensor: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """

    gr = torch.empty(lrot.shape, device=lrot.device, dtype=lrot.dtype)
    gr[..., :1, :, :] = lrot[..., :1, :, :]

    gp = torch.empty(lpos.shape, device=lpos.device, dtype=lpos.dtype)
    gp[..., :1, :] = lpos[..., :1, :]

    for joint_idx in range(1, parents.shape[-1]):
        parent_joint = parents[..., joint_idx]
        empty_channels: torch.Tensor = parent_joint < 0

        joint_mask: torch.Tensor = ~empty_channels
        actual_parents = parent_joint[joint_mask]

        gr_parent = (
            gr.swapaxes(1, 2)
            [torch.arange(joint_mask.shape[0], device=gp.device)[joint_mask], actual_parents][..., None, :, :]
        )

        gp_parent = (
            gp.swapaxes(1, 2)
            [torch.arange(joint_mask.shape[0], device=gp.device)[joint_mask], actual_parents][..., None, :]
        )

        gr_i = torch.matmul(gr_parent, lrot[..., joint_idx:joint_idx + 1, :, :][joint_mask])
        gp_i = gp_parent + \
               torch.matmul(gr_parent, lpos[..., joint_idx:joint_idx + 1, :, None][joint_mask]).squeeze(-1)

        gr[..., joint_idx:joint_idx + 1, :, :][joint_mask] = gr_i
        gp[..., joint_idx:joint_idx + 1, :][joint_mask] = gp_i

    return gr, gp


# Credit to Motion In-betweening via Two-stage Transformers
def to_start_centered_data_local(positions, rotations, parents,
                                 context_len=1,
                                 forward_axis="x", root_idx=0, return_offset=False):
    """
    Center raw data at the start of transition.
    Last context frame is moved to origin (only x and z axis, y unchanged)
    and facing forward_axis.

    Args:
        positions (tensor): (..., seq, joint, 3), raw position data
        rotations (tensor): (..., seq, joint, 3, 3), raw rotation data
        context_len (int): length of context frames
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.
        return_offset (bool): If True, return root position and rotation
            offset as well.
    Returns:
        If return_offset == False:
        (tensor, tensor): (new position, new rotation), shape same as input
        If return_offset == True:
        (tensor, tensor, tensor, tensor):
            (new positions, new rotations, root pos offset, root rot offset)
    """
    pos = positions.clone().detach()
    rot = rotations.clone().detach()
    frame = context_len - 1

    with torch.no_grad():
        # root position on xz axis at last context frame as position offset
        root_pos_offset = pos[..., frame:frame + 1, root_idx, :]  # Y coord must be replaced!
        root_pos_offset = root_pos_offset.clone().detach()
        
        # Correct height so that skeleton doesn't float
        _, g_pos = fk(rot, pos, parents)
        min_foot_level = __get_min_y_level(g_pos, ref_frame=frame)
        root_pos_offset[..., 1] = min_foot_level
        
        pos = _apply_root_pos_offset(pos, root_pos_offset, root_idx)

        # last context frame root rotation as rotation offset
        root_rot_offset = _get_root_rot_offset_at_frame(
            pos, rot, frame, forward_axis, root_idx)
        # pos, rot = _apply_root_rot_offset_local(pos, rot, root_rot_offset, root_idx)

    if return_offset:
        return pos.detach(), rot.detach(), root_pos_offset, root_rot_offset
    else:
        return pos.detach(), rot.detach()


def euler_to_matrix9D_torch(euler, order="zyx", unit="degrees"):
    """
    Euler angle to 3x3 rotation matrix.

    Args:
        euler (tensor): Euler angle. Shape: (..., 3)
        order (str, optional):
            Euler rotation order AND order of parameter euler.
            E.g. "yxz" means parameter "euler" is (y, x, z) and
            rotation order is z applied first, then x, finally y.
            i.e. p' = YXZp, where p' and p are column vectors.
            Defaults to "zyx".
    """
    dtype = euler.dtype
    device = euler.device

    mat = torch.eye(3, dtype=dtype, device=device)

    if unit == "degrees":
        euler_radians = euler / 180.0 * np.pi
    elif unit == "radians":
        euler_radians = euler
    else:
        raise ValueError("Invalid unit value. Given: {},"
                         "supports: degrees or radians.".format(unit))

    for idx, axis in enumerate(order):
        angle_radians = euler_radians[..., idx:idx + 1]

        # shape: (..., 1)
        sin = torch.sin(angle_radians)
        cos = torch.cos(angle_radians)

        ones = torch.ones(sin.shape, dtype=dtype, device=device)
        zeros = torch.zeros(sin.shape, dtype=dtype, device=device)

        if axis == "x":
            # shape(..., 9)
            rot_mat = torch.cat([
                ones, zeros, zeros,
                zeros, cos, -sin,
                zeros, sin, cos], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        elif axis == "y":
            # shape(..., 9)
            rot_mat = torch.cat([
                cos, zeros, sin,
                zeros, ones, zeros,
                -sin, zeros, cos], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        else:
            # shape(..., 9)
            rot_mat = torch.cat([
                cos, -sin, zeros,
                sin, cos, zeros,
                zeros, zeros, ones], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        mat = torch.matmul(mat, rot_mat)

    return mat


def _apply_root_pos_offset(pos, root_pos_offset, root_idx=0):
    """
    Apply root joint position offset.

    Args:
        pos (tensor): (..., seq, joint, 3)
        root_pos_offset (tensor): (..., seq, 3)
        root_idx (int, optional): Root joint index. Defaults to 0.

    Returns:
        tensor: new pos, shape same as input pos
    """
    pos[..., :, root_idx, :] = pos[..., :, root_idx, :] - root_pos_offset
    return pos


def _apply_root_rot_offset_local(pos, rot, root_rot_offset, root_idx=0):
    """
    Apply root rotation offset

    Args:
        pos (tensor): (..., seq, joint, 3)
        rot ([type]): (..., seq, joint, 3, 3)
        root_rot_offset: (..., seq, 3, 3)
        root_idx (int, optional): [description]. Defaults to 0.

    Returns:
        (tensor, tensor): new pos, new rot. Shape same as input.
    """
    rot[..., :, root_idx, :, :] = torch.matmul(
        root_rot_offset, rot[..., :, root_idx, :, :])
    pos[..., :, root_idx, :] = torch.matmul(
        root_rot_offset, pos[..., :, root_idx, :, None]).squeeze(-1)

    return pos, rot


def _get_root_rot_offset_at_frame(pos, rot, frame,
                                  forward_axis="x", root_idx=0):
    """
    Get the rotation offset that makes root joint faces forward_axis at
    given frame.

    Args:
        pos (tensor): (..., seq, joint, 3),
        rot (tensor): (..., seq, joint, 3, 3)
        frame (int): frame index
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.

    Raises:
        ValueError: if forward_axis is given an invalid value

    Returns:
        tensor: (..., seq, 3, 3), root rotation offset
    """
    dtype = pos.dtype
    device = pos.device

    root_rot = rot[..., frame:frame + 1, root_idx, :, :]

    # y axis is the local forward axis for root joint
    # We want to make root's local y axis after rotation,
    # align with world forward_axis
    y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    y = y.repeat(*root_rot.shape[:-2], 1)
    y_rotated = torch.matmul(root_rot, y[..., None]).squeeze(-1)
    y_rotated[..., 1] = 0  # project to xz-plane
    y_rotated = _normalize_torch(y_rotated)

    if forward_axis == "x":
        forward = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    elif forward_axis == "z":
        forward = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    elif forward_axis == "-z":
        forward = torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=device)
    else:
        raise ValueError("forward_axis expect value 'x' or 'z', "
                         "got '{}'.".format(forward_axis))

    forward = forward.repeat(*root_rot.shape[:-2], 1)

    dot = _batch_vector_dot_torch(y_rotated, forward)
    cross = torch.cross(y_rotated, forward)
    angle = torch.atan2(_batch_vector_dot_torch(cross, y), dot)

    zeros = torch.zeros(angle.shape, dtype=dtype, device=device)
    euler_angle = torch.cat([zeros, angle, zeros], dim=-1)
    root_rot_offset = euler_to_matrix9D_torch(euler_angle, unit="radians")

    return root_rot_offset.detach()


def _normalize_torch(tensor, dim=-1, eps=1e-5):
    """
    Normalize tensor along given dimension.

    Args:
        tensor (Tensor): Tensor.
        dim (int, optional): Dimension to normalize. Defaults to -1.
        eps (float, optional): Small value to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        Tensor: Normalized tensor.
    """
    return F.normalize(tensor, p=2, dim=dim, eps=eps)


def _batch_vector_dot_torch(vec1, vec2):
    """
    Batch vector dot product.
    """
    dot = torch.matmul(vec1[..., None, :], vec2[..., None])
    return dot.squeeze(-1)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _remove_quat_discontinuities(rotations):
    """
    Removing quat discontinuities on the time dimension (removing flips)
    Note: this function cannot be back propagated.

    Args:
        rotations (tensor): Shape: (..., seq, joint, 4)

    Returns:
        tensor: The processed tensor without quaternion inversion.
    """
    with torch.no_grad():
        rots_inv = -rotations

        for i in range(1, rotations.shape[-3]):
            # Compare dot products
            prev_rot = rotations[..., i - 1, :, :]
            curr_rot = rotations[..., i, :, :]
            curr_inv_rot = rots_inv[..., i, :, :]
            replace_mask = (
                    torch.sum(prev_rot * curr_rot, dim=-1, keepdim=True) <
                    torch.sum(prev_rot * curr_inv_rot, dim=-1, keepdim=True)
            )
            rotations[..., i, :, :] = (
                    replace_mask * rots_inv[..., i, :, :] +
                    replace_mask.logical_not() * rotations[..., i, :, :]
            )

    return rotations


# FIXME: should only be used at whole animation level (TODO)
def __get_min_y_level(g_pos: torch.Tensor, ref_frame: int) -> torch.Tensor:
    return g_pos[..., ref_frame:ref_frame + 1, :, 1].min(dim=2)[0]

