import torch


def torch_mdc_mdcss_metrics(g_pos: torch.Tensor, n: int, bone_lengths: torch.Tensor, fps: int,
                            noise_windows=None, jitter_std=0.0125, fps_div=3,
                            use_hamm_window=False, **kwargs):
    with torch.no_grad():
        if noise_windows is not None:
            noise_scale = jitter_std * bone_lengths.sum(dim=-1)
            for window_start, window_end in noise_windows:
                jittery_joint = (bone_lengths > 0.0001).long().float().multinomial(1)
                g_pos[:, window_start:window_end, jittery_joint, :] = torch.normal(
                    mean=g_pos[:, window_start:window_end, jittery_joint, :],
                    std=noise_scale * torch.ones(g_pos[:, window_start:window_end, jittery_joint, :].shape,
                                                 device=g_pos.device))

        g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
        vel_cosines = (g_vel[..., 1:, :, :] * g_vel[..., :-1, :, :]).sum(dim=-1)
        norms = g_vel[..., 1:, :, :].norm(dim=-1) * g_vel[..., :-1, :, :].norm(dim=-1)
        norms[norms == 0.] = 1
        vel_cosines = vel_cosines / norms
        vel_angles = torch.acos(vel_cosines.clamp(-1, 1))
        acc_norms = (g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]).norm(dim=-1)
        per_joint_metric = fps * vel_angles.pow(n) * acc_norms / bone_lengths.sum(dim=-1)
        sliding_window_frames = int(fps / fps_div)  # 1/3 second of window by defualt
        mdc_metric = per_joint_metric.max(dim=-1).values
        window_metric = mdc_metric.unfold(1, sliding_window_frames, 1)
        if use_hamm_window:
            window_metric *= torch.hamming_window(
                sliding_window_frames,
                device=window_metric.device,
                dtype=window_metric.dtype,
            )
        fft_vals_windowed = torch.fft.fft(window_metric, dim=-1).real
        mdcss_metric = fft_vals_windowed[..., 1:].max(dim=-1).values
        return mdc_metric, mdcss_metric


def torch_mdcss(
        g_pos: torch.Tensor,
        bone_lengths: torch.Tensor,
        fps: int,

        fps_div: int = 3,
        n: int = 2,
        use_hamm_window=False,
        **kwargs
):
    with torch.no_grad():
        g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
        vel_cosines = (g_vel[..., 1:, :, :] * g_vel[..., :-1, :, :]).sum(dim=-1)
        norms = g_vel[..., 1:, :, :].norm(dim=-1) * g_vel[..., :-1, :, :].norm(dim=-1)
        norms[norms == 0.] = 1
        vel_cosines = vel_cosines / norms
        vel_angles = torch.acos(vel_cosines.clamp(-1, 1))
        acc_norms = (g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]).norm(dim=-1)
        per_joint_metric = fps * vel_angles.pow(n) * acc_norms / bone_lengths.sum(dim=-1)
        sliding_window_frames = int(fps / fps_div)  # 1/3 second of window by defualt
        base_metric = per_joint_metric.max(dim=-1).values
        window_metric = base_metric.unfold(1, sliding_window_frames, 1)
        if use_hamm_window:
            window_metric *= torch.hamming_window(
                sliding_window_frames,
                device=window_metric.device,
                dtype=window_metric.dtype,
            )
        fft_vals_windowed = torch.fft.fft(window_metric, dim=-1).real
        mdcss_metric = fft_vals_windowed[..., 1:].max(dim=-1).values
        return mdcss_metric


def __extract_foot_vel(g_pos, foot_joint_idx):
    # gpos: global position, (batch, seq, joint, 3)

    foot_vel = (
            g_pos[..., 1:, foot_joint_idx, :] -
            g_pos[..., :-1, foot_joint_idx, :]
    )

    # Pad zero on the first frame for shape consistency
    zeros_shape = list(foot_vel.shape)
    zeros_shape[-3] = 1
    zeros = torch.zeros(
        zeros_shape, device=foot_vel.device, dtype=foot_vel.dtype)
    foot_vel = torch.cat([zeros, foot_vel], dim=-3)

    return foot_vel
