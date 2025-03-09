import numpy as np


def scale_skeletons(lrot, lpos):
    pass    # TODO


# Credit to Motion In-betweening via Two-stage Transformers
def fk(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (ndarray): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (ndarray): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D ndarray): Parent indices.

    Returns:
        ndarray, ndarray: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = np.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
               np.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)


# Credit to Motion In-betweening via Two-stage Transformers
def extract_feet_contacts(global_pos, lfoot_idx, rfoot_idx,
                          vel_threshold=0.2):
    """
    Extracts binary tensors of feet contacts.

    Args:
        global_pos (ndarray): Global positions of joints.
            Shape: (frames, joints, 3)
        lfoot_idx (int): Left foot joints indices.
        rfoot_idx (int): Right foot joints indices.
        vel_threshold (float, optional): Velocity threshold to consider a
            joint as stationary. Defaults to 0.2.

    Returns:
        ndarray: Binary ndarray indicating left and right foot's contact to
            the ground. Shape: (frames, len(lfoot_idx) + len(rfoot_idx))
    """
    lfoot_vel = np.abs(global_pos[1:, lfoot_idx, :] -
                       global_pos[:-1, lfoot_idx, :])
    rfoot_vel = np.abs(global_pos[1:, rfoot_idx, :] -
                       global_pos[:-1, rfoot_idx, :])

    contacts_l = (np.sum(lfoot_vel, axis=-1) < vel_threshold)
    contacts_r = (np.sum(rfoot_vel, axis=-1) < vel_threshold)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r
