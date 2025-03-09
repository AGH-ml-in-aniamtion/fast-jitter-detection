import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

from analysis.dataset_analyzer import DatasetConfigurator
from utils.torch_utils import torch_anim
from utils.torch_utils.torch_anim import to_start_centered_data_local


def fit_to_train_pca(dataset_analyzer: DatasetConfigurator, n_components: int = 22, use_acc: bool = True):
    random_state = 123
    pca = PCA(n_components, random_state=random_state)
    train_samples = []

    fps = dataset_analyzer.dataset_metadata.fps
    fps_div = 3
    sliding_window_frames = int(fps / fps_div)  # 1/3 second of window by default

    print("Preparing data for PCA...")
    for data in dataset_analyzer.train_loader:
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, sequence_name, data_idx) = data

        with torch.no_grad():
            sample_parents = parents[0]
            centered_lpos, centered_lrot = to_start_centered_data_local(lpos, lrot, sample_parents)
            _, g_pos = torch_anim.fk(centered_lrot, centered_lpos, sample_parents)

            g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
            if use_acc:
                g_acc = g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]
                metric = g_acc
            else:
                metric = g_vel

            metric *= fps / bone_lengths.sum(dim=-1)
            window_metric = metric.unfold(1, sliding_window_frames, 1).cpu().numpy()

        train_samples.append(window_metric)

    train_data = np.concatenate(train_samples, axis=1)
    train_data = train_data.reshape((train_data.shape[1], -1))  # TODO: This might need to be changed

    print("Fitting PCA...")
    train_transformed = pca.fit_transform(train_data)
    print("PCA ready for evaluation")
    train_reconstructed = pca.inverse_transform(train_transformed)
    per_joint_error = np.sum(np.power((train_data - train_reconstructed), 2).reshape(1, -1, 22, 3), axis=-1)
    max_rec_error = np.max(per_joint_error)
    extra_kwargs = {
        'error_threshold': 1.5 * max_rec_error,
    }
    return pca, extra_kwargs


def pca_eval(
        g_pos: torch.Tensor,
        bone_lengths: torch.Tensor,
        fps: int,

        fps_div: int = 3,
        use_acc: bool = True,
        model: PCA = None,

        **kwargs,
):
    g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
    if use_acc:
        g_acc = g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]
        metric = g_acc
    else:
        metric = g_vel

    sliding_window_frames = fps // fps_div  # 1/3 sec
    metric *= fps / bone_lengths.sum(dim=-1)
    window_metric = metric.unfold(1, sliding_window_frames, 1).cpu().numpy()
    window_metric = window_metric.reshape((window_metric.shape[1], -1))  # TODO: This might need to be changed

    transformed_data = model.transform(window_metric)
    rec_window_metric = model.inverse_transform(transformed_data)

    torch_window_metric = torch.from_numpy(window_metric)
    torch_rec_window_metric = torch.from_numpy(rec_window_metric)

    per_joint_error = torch.sum(
        torch.pow((torch_window_metric - torch_rec_window_metric), 2).reshape(1, -1, 10, 22, 3),
        dim=-1
    )
    per_window_metric = torch.amax(per_joint_error, dim=(2, 3))
    return per_window_metric


def pca_pred(rec_error, noise_windows, sliding_window_frames, error_threshold: float, **kwargs):
    # mean_rec_error = rec_error.mean()

    pred_frame_labels_batches = []
    prev_end = 0
    sliding_window_frames -= 1
    for start, end in noise_windows:
        corrected_start, corrected_end = start - sliding_window_frames, end - sliding_window_frames
        pred_frame_labels_batches.append(torch.where(rec_error[:, prev_end:corrected_start] > error_threshold, 1, 0))
        pred_frame_labels_batches.append(
            torch.where(rec_error[:, corrected_start:corrected_end] > error_threshold, 1, 0).amax(dim=-1)[..., None])
        prev_end = end
    pred_frame_labels_batches.append(
        torch.where(rec_error[:, prev_end - sliding_window_frames:] > error_threshold, 1, 0))
    return torch.concatenate(pred_frame_labels_batches, dim=1)


def savgol_eval(
        g_pos: torch.Tensor,
        bone_lengths: torch.Tensor,
        fps: int,

        fps_div: int = 3,
        **kwargs,
):
    g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
    g_acc = g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]
    metric = g_acc

    sliding_window_frames = fps // fps_div  # 1/3 sec
    metric *= fps / bone_lengths.sum(dim=-1)
    # window_metric = metric.unfold(1, sliding_window_frames, 1).cpu().numpy()
    per_frame_values = metric.reshape((metric.shape[1], -1))

    filtered_frames = savgol_filter(per_frame_values.cpu().numpy(), sliding_window_frames, 2, axis=0)
    filtered_frames = torch.from_numpy(filtered_frames).to(device=per_frame_values.device)
    rec_frame_metric = (filtered_frames - per_frame_values) ** 2
    rec_window_metric = rec_frame_metric.unfold(0, sliding_window_frames, 1).reshape(-1, 10, 22, 3)
    per_joint_error = torch.sum(rec_window_metric, dim=-1)
    per_window_metric = torch.amax(per_joint_error, dim=(1, 2))
    return per_window_metric[None, ...]


def savgol_pred(rec_error, noise_windows, sliding_window_frames, **kwargs):
    error_threshold = 5.5  # 7 * rec_error.mean() # 20 for sum

    pred_frame_labels_batches = []
    prev_end = 0
    sliding_window_frames -= 1
    for start, end in noise_windows:
        corrected_start, corrected_end = start - sliding_window_frames, end - sliding_window_frames
        pred_frame_labels_batches.append(torch.where(rec_error[:, prev_end:corrected_start] > error_threshold, 1, 0))
        pred_frame_labels_batches.append(
            torch.where(rec_error[:, corrected_start:corrected_end] > error_threshold, 1, 0).amax(dim=-1)[..., None])
        prev_end = end
    pred_frame_labels_batches.append(
        torch.where(rec_error[:, prev_end - sliding_window_frames:] > error_threshold, 1, 0))
    return torch.concatenate(pred_frame_labels_batches, dim=1)


def prepare_data_acc(data_loader, sliding_window_frames: int, fps: int):
    x_samples = []
    y_samples = []
    for data in data_loader:
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, sequence_name, data_idx) = data

        with torch.no_grad():
            sample_parents = parents[0]
            centered_lpos, centered_lrot = to_start_centered_data_local(lpos, lrot, sample_parents)
            _, g_pos = torch_anim.fk(centered_lrot, centered_lpos, sample_parents)

            g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
            g_acc = g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]
            metric = g_acc

            metric *= fps / bone_lengths.sum(dim=-1)
            window_metric = metric.unfold(1, sliding_window_frames + 1, 1)

        x_samples.append(window_metric[..., :-1])
        y_samples.append(window_metric[..., -1:])

    x_data = torch.concatenate(x_samples, dim=1)[0]
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[-1], -1)
    y_data = torch.concatenate(y_samples, dim=1)[0]
    y_data = y_data.reshape(y_data.shape[0], -1)
    return x_data, y_data
