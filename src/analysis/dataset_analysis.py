import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from analysis.dataset_analyzer import DatasetConfigurator
from data_models.parsers import bvh_parser
from metrics.dataset_metrics import torch_mdc_mdcss_metrics
from utils.torch_utils import torch_anim
from utils.torch_utils.torch_anim import to_start_centered_data_local



def calculate_metrics(dataset_analyzer: DatasetConfigurator, n: int = 2, noise_windows=None, fps_div=3,
                      use_hamm_window=False, **kwargs):
    acc_metric_batches = []
    metric_std_batches = []
    error_int_cnt, warning_int_cnt = 0, 0
    error_frame_cnt, warning_frame_cnt = 0, 0
    total_frames = 0
    total_fft_windows = 0
    for i, data in enumerate(dataset_analyzer.train_loader, 0):
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, sequence_name, data_idx) = data
        sample_parents = parents[0]
        centered_lpos, centered_lrot = to_start_centered_data_local(lpos, lrot, sample_parents)
        centered_rot_global, centered_pos_global = torch_anim.fk(centered_lrot, centered_lpos, sample_parents)
        acc_metric, fft_metric = torch_mdc_mdcss_metrics(
            centered_pos_global, n,
            bone_lengths=bone_lengths,
            fps=dataset_analyzer.dataset_metadata.fps,
            noise_windows=noise_windows,
            fps_div=fps_div,
            use_hamm_window=use_hamm_window,
        )
        acc_metric_batches.append(acc_metric)
        metric_std_batches.append(fft_metric)
        total_frames += centered_pos_global.shape[1]
        total_fft_windows += fft_metric.shape[1]

        if kwargs.get("analyze_errors_and_warnings"):
            error_intervals, warning_intervals, ef, wf, _ = __analyze_errors_and_warnings(fft_metric[0])
            error_frame_cnt += ef
            warning_frame_cnt += wf
            error_int_cnt += len(error_intervals)
            warning_int_cnt += len(warning_intervals)
            if ef > 0:
                print(f"Errors on sequence: {sequence_name[0]} ({ef} errors)")

    dataset_mdc_metric = sum([m.sum() for m in acc_metric_batches]) / total_frames
    dataset_fft_metric = sum([m.sum() for m in metric_std_batches]) / total_fft_windows
    print(f"MDC metric (averaged over joints and then over frames): {dataset_mdc_metric}")
    print(f"MDCSS metric (averaged over joints and then over frames): {dataset_fft_metric}")
    if kwargs.get("analyze_errors_and_warnings"):
        print(f"Error intervals: {error_int_cnt}")
        print(f"Error frames: {error_frame_cnt}")
        print(f"Warning intervals: {warning_int_cnt}")
        print(f"Warning frames: {warning_frame_cnt}")
    return range(acc_metric_batches[0].shape[1]), acc_metric_batches, metric_std_batches, f"metric_acc_acos_based_{n}"


def subsample_clip(dataset_analyzer: DatasetConfigurator, sample_every: int, output_file: str,
                   ref_skel: str = "data/datasets/bvh/Sophie_skel.bvhskel"):
    for j, data in enumerate(dataset_analyzer.train_loader, 0):
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, data_idx) = data

        subsampled_lpos, subsampled_lrot = lpos[:, ::sample_every, ...], lrot[:, ::sample_every, ...]
        euler_lrot = torch_anim.matrix9D_to_euler_angles(subsampled_lrot)
        bvh_parser.save_bvh(
            ref_skel,
            subsampled_lpos[j],
            euler_lrot[j],
            joint_offsets[j],
            output_file
        )


def calculate_metrics_with_random_noise(dataset_analyzer: DatasetConfigurator, n: int = 2, jitter_std=0.0075, **kwargs):
    acc_metric_batches = []
    metric_std_batches = []
    error_cnt, warning_cnt = 0, 0
    total_frames = 0
    total_fft_windows = 0

    total_noise_windows, detected_noise_windows = 0, 0

    for i, data in enumerate(dataset_analyzer.train_loader, 0):
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, sequence_name, data_idx) = data

        noise_windows = []
        frame_count = lpos.shape[1]
        fps = dataset_analyzer.dataset_metadata.fps
        single_window_range = 10 * fps
        samples = frame_count // single_window_range  # 1 window per 10 sec
        noise_window_starts = np.random.choice(single_window_range, size=samples, replace=False)
        noise_window = np.zeros(frame_count)
        for j, noise_start in enumerate(noise_window_starts):
            interval_start = j * single_window_range
            noise_length = np.random.randint(fps // 4, 2 * fps)
            end_on_interval = interval_start + min(single_window_range, noise_start + noise_length)
            window_noise_start = interval_start + noise_start
            window_noise_end = min(frame_count, end_on_interval)
            noise_window[window_noise_start:window_noise_end] = 1
            noise_windows.append((window_noise_start, window_noise_end))

        sample_parents = parents[0]
        centered_lpos, centered_lrot = to_start_centered_data_local(lpos, lrot, sample_parents)
        centered_rot_global, centered_pos_global = torch_anim.fk(centered_lrot, centered_lpos, sample_parents)
        acc_metric, fft_metric = torch_mdc_mdcss_metrics(centered_pos_global, n,
                                                         bone_lengths=bone_lengths,
                                                         fps=dataset_analyzer.dataset_metadata.fps,
                                                         noise_windows=noise_windows,
                                                         jitter_std=jitter_std)
        acc_metric_batches.append(acc_metric)
        metric_std_batches.append(fft_metric)
        total_frames += centered_pos_global.shape[1]
        total_fft_windows += fft_metric.shape[1]

        if kwargs.get("analyze_errors_and_warnings"):
            if kwargs.get("analyze_noise_windows"):
                total_noise_windows += len(noise_windows)
            error_intervals, warning_intervals, ef, wf, dw = __analyze_errors_and_warnings(fft_metric[0], noise_windows)
            error_cnt += len(error_intervals)
            warning_cnt += len(warning_intervals)
            detected_noise_windows += dw

    dataset_acc_metric = sum([m.sum() for m in acc_metric_batches]) / total_frames
    dataset_fft_metric = sum([m.sum() for m in metric_std_batches]) / total_fft_windows
    print(f"MDC metric (averaged over joints and then over frames): {dataset_acc_metric}")
    print(f"MDCSS metric (averaged over joints and then over frames): {dataset_fft_metric}")
    if kwargs.get("analyze_errors_and_warnings"):
        print(f"Error windows: {error_cnt}")
        print(f"Warning windows: {warning_cnt}")
    if kwargs.get("analyze_noise_windows"):
        print(f"Detected windows: {detected_noise_windows}/{total_noise_windows}")
    return detected_noise_windows / total_noise_windows, detected_noise_windows, total_noise_windows


def __analyze_errors_and_warnings(fft_metric, noise_windows=None):
    warning_intervals = []
    error_intervals = []
    error_frames, warning_frames = 0, 0
    windows_detected = 0

    interval_start, interval_level = -1, 0
    for j, val in enumerate(fft_metric):
        if val < 8:
            if interval_level == 1:
                interval_end = j
                warning_intervals.append((interval_start, interval_end))
                interval_start = j
            elif interval_level == 2:
                interval_end = j
                error_intervals.append((interval_start, interval_end))
                interval_start = j
            interval_level = 0
        if 8 <= val <= 20:
            warning_frames += 1
            if interval_level == 0:
                interval_start = j
            elif interval_level == 2:
                interval_end = j
                error_intervals.append((interval_start, interval_end))
                interval_start = j
            interval_level = 1
        if val >= 20:
            error_frames += 1
            windows_detected += __update_detected_windows(j, noise_windows)
            if interval_level == 0:
                interval_start = j
            elif interval_level == 1:
                interval_end = j
                warning_intervals.append((interval_start, interval_end))
                interval_start = j
            interval_level = 2
    return error_intervals, warning_intervals, error_frames, warning_frames, windows_detected


def __update_detected_windows(j, noise_windows: list):
    if not noise_windows:
        return 0
    window_idx = None
    for i, (start, end) in enumerate(noise_windows):
        if start <= j < end:
            window_idx = i
            break
        if j < start:
            break
    if window_idx is not None:
        noise_windows.pop(window_idx)
        return 1
    return 0


def evaluate_custom_jitter(
        dataset_analyzer: DatasetConfigurator,
        metric_evaluator: callable,
        pred_evaluator: callable,
        n: int = 2,
        jitter_std: float = 0.0075,
        verbose=False,
        custom_noise_windows=None,
        run_smoke_test=False,
        **kwargs,
):
    metric_batches = []
    window_true_classification = []
    window_pred_classification = []
    total_fft_windows = 0

    excluded_sequences = ['obstacles5_subject3.bvh']

    for data in dataset_analyzer.test_loader:
        (lpos, lrot, joint_offsets,
         foot_contact, parents, bone_lengths, sequence_name, data_idx) = data


        fps = dataset_analyzer.dataset_metadata.fps
        sliding_window_frames = int(fps / 3)  # 1/3 second of window by default
        noise_windows = []
        frame_count = lpos.shape[1]
        single_window_range = 10 * fps
        max_window_length = fps
        potential_jitter_start = sliding_window_frames

        samples = frame_count // single_window_range  # 1 window per 10 sec
        noise_window_starts = np.random.choice(
            single_window_range - max_window_length - potential_jitter_start,
            size=samples,
            replace=True,
        )
        noise_window = np.zeros(frame_count)

        for j, noise_start in enumerate(noise_window_starts):
            interval_start = j * single_window_range
            noise_length = np.random.randint(fps // 4, max_window_length)
            end_on_interval = interval_start + potential_jitter_start + min(single_window_range, noise_start + noise_length)
            window_noise_start = interval_start + noise_start + potential_jitter_start
            window_noise_end = min(frame_count, end_on_interval)
            noise_window[window_noise_start:window_noise_end] = 1
            noise_windows.append((window_noise_start, window_noise_end))

        if sequence_name[0] in excluded_sequences:
            # Special treatment of already jittery sequences
            # Error value at frames: (3941, 3952)
            # Error value at frames: (4696, 4697)
            # Error value at frames: (5703, 5704)
            # Error value at frames: (6537, 6538)
            # Error value at frames: (7022, 7023)
            # Error value at frames: (7027, 7028)
            if not run_smoke_test:
                continue
            else:
                noise_windows = [
                    (3931, 3962),
                    (4686, 4707),
                    (5693, 5714),
                    (6527, 6548),
                    (7012, 7038),
                ]
                noise_window = np.zeros(frame_count)
                for start, end in noise_windows:
                    noise_window[start:end] = 1

        sample_parents = parents[0]
        centered_lpos, centered_lrot = to_start_centered_data_local(lpos, lrot, sample_parents)
        centered_rot_global, centered_pos_global = torch_anim.fk(centered_lrot, centered_lpos, sample_parents)

        if sequence_name[0] not in excluded_sequences:
            noise_scale = jitter_std * bone_lengths.sum(dim=-1)
            for potential_jitter_start, window_end in noise_windows:
                jittery_joint = (bone_lengths > 0.0001).long().float().multinomial(1)
                centered_pos_global[:, potential_jitter_start:window_end, jittery_joint, :] = torch.normal(
                    mean=centered_pos_global[:, potential_jitter_start:window_end, jittery_joint, :],
                    std=noise_scale * torch.ones(centered_pos_global[:, potential_jitter_start:window_end, jittery_joint, :].shape,
                                                 device=centered_pos_global.device))

        metric_value = metric_evaluator(
            centered_pos_global,
            bone_lengths=bone_lengths,
            fps=dataset_analyzer.dataset_metadata.fps,
        )
        jitter_aggregated_metric = metric_reduce_on_jitter_windows(metric_value, noise_windows, sliding_window_frames)
        metric_batches.append(jitter_aggregated_metric)
        total_fft_windows += metric_value.shape[1]


        window_classification_batch = __get_window_labels_reduced(centered_pos_global, noise_windows, sliding_window_frames)
        window_true_classification.append(window_classification_batch[:, :-2])

        pred_class_batch = pred_evaluator(
            metric_value,
            noise_windows=noise_windows,
            sliding_window_frames=sliding_window_frames,
            **kwargs,
        )
        window_pred_classification.append(pred_class_batch)

        if sequence_name[0] in excluded_sequences and run_smoke_test:
            print(f"Smoke test detection result: "
                  f"{pred_class_batch.sum().item()} errors detected")

            expected_indices = [
                (3915, 3930),
                (4630, 4645),
                (5608, 5623),
                (6414, 6429),
                (6870, 6885),
            ]

            correct_detections = 0
            for start, end in expected_indices:
                res = "detected" if pred_class_batch[0, start:end].max() == 1.0 else "not detected"
                print(f"Expected following window: [{start}, {end}] => {res}")
                correct_detections += pred_class_batch[0, start:end].sum()
            print(f"Incorrect detections: {pred_class_batch.sum() - correct_detections}")
            print(f"All detections: {(pred_class_batch[0] == pred_class_batch.max()).nonzero().cpu().numpy()}")

    # dataset_acc_metric = sum([m.sum() for m in acc_metric_batches]) / total_frames

    # avg_dataset_metric = sum([m.sum() for m in metric_batches]) / total_fft_windows
    dataset_metrics_per_window = torch.concat(metric_batches, dim=1).flatten()
    dataset_window_true_classification = torch.concat(window_true_classification, dim=1).flatten()
    dataset_window_pred_classification = torch.concat(window_pred_classification, dim=1).flatten()

    # print(f"MDC metric (averaged over joints and then over frames): {dataset_acc_metric}")
    # print(f"MDCSS metric (averaged over joints and then over frames): {dataset_fft_metric}")

    roc_auc = roc_auc_score(
        dataset_window_true_classification.cpu().numpy(),
        dataset_metrics_per_window.cpu().numpy(),
    )
    # f1_res = f1_score(
    #     dataset_window_true_classification.cpu().numpy(),
    #     dataset_window_pred_classification.cpu().numpy(),
    # )
    class_report = classification_report(
        dataset_window_true_classification.cpu().numpy(),
        dataset_window_pred_classification.cpu().numpy(),
        output_dict=True,
    )

    if verbose:
        # print(f"Avg metric: {avg_dataset_metric}")
        print(f"AUC ROC score: {roc_auc}")
        print(f"F1 score: {class_report['1.0']['f1-score']}")
        print(f"Sensitivity (TPR): {class_report['1.0']['recall']}")
    # print(f"Error windows: {error_cnt}")
    # print(f"Warning windows: {warning_cnt}")
    # print(f"Detected windows: {detected_noise_windows}/{total_noise_windows}")
    # return detected_noise_windows / total_noise_windows, detected_noise_windows, total_noise_windows
    return roc_auc, class_report['1.0']['f1-score'], class_report['1.0']['recall'], class_report['1.0']['precision']


def __get_window_labels_reduced(g_pos, noise_windows, sliding_window_frames):
    frame_labels_batches = []
    zero_labels = torch.zeros(*g_pos.shape[:2], device=g_pos.device)
    prev_end = 0
    sliding_window_frames -= 1
    for start, end in noise_windows:
        frame_labels_batches.append(zero_labels[:, prev_end:start - sliding_window_frames])
        frame_labels_batches.append(torch.ones((g_pos.shape[0], 1), device=g_pos.device))
        prev_end = end
    frame_labels_batches.append(zero_labels[:, prev_end - sliding_window_frames:-sliding_window_frames])

    windowed_frame_labels = torch.concatenate(frame_labels_batches, dim=1)
    return windowed_frame_labels


# def __get_window_labels_raw(g_pos, noise_windows, sliding_window_frames):
#     frame_labels = torch.zeros(*g_pos.shape, device=g_pos.device)
#     for start, end in noise_windows:
#         frame_labels[:, start:end, ...] = 1
#     jitter_frames_per_window = frame_labels.unfold(1, sliding_window_frames, 1).amax(dim=(2, 3)).sum(dim=-1)
#     windowed_frame_labels = torch.where(jitter_frames_per_window >= 5.0, 1, 0)
#     return windowed_frame_labels


def mdcss_classify(fft_metric, noise_windows, sliding_window_frames, **kwargs):
    pred_frame_labels_batches = []
    prev_end = 0
    sliding_window_frames -= 1
    for start, end in noise_windows:
        corrected_start, corrected_end = start - sliding_window_frames, end - sliding_window_frames
        pred_frame_labels_batches.append(torch.where(fft_metric[:, prev_end:corrected_start] > 20, 1, 0))
        pred_frame_labels_batches.append(torch.where(fft_metric[:, corrected_start:corrected_end] > 20, 1, 0).amax(dim=-1)[..., None])
        prev_end = end
    pred_frame_labels_batches.append(torch.where(fft_metric[:, prev_end - sliding_window_frames:] > 20, 1, 0))
    return torch.concatenate(pred_frame_labels_batches, dim=1)


def metric_reduce_on_jitter_windows(metric, noise_windows, sliding_window_frames):
    pred_score_batches = []
    prev_end = 0
    sliding_window_frames -= 1
    for start, end in noise_windows:
        corrected_start, corrected_end = start - sliding_window_frames, end - sliding_window_frames
        pred_score_batches.append(metric[:, prev_end:corrected_start])
        pred_score_batches.append(metric[:, corrected_start:corrected_end].amax(dim=-1)[..., None])
        prev_end = end
    pred_score_batches.append(metric[:, prev_end - sliding_window_frames:])
    return torch.concatenate(pred_score_batches, dim=1)

