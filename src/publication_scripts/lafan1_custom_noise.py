import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

FONT_SIZE = 20
matplotlib.rcParams.update({'font.size': FONT_SIZE})

if __name__ == "__main__":
    plots_per_row = 1
    figure, axis = plt.subplots(3, plots_per_row, constrained_layout=True, figsize=(10, 20))

    plot_y_data, plot_y_std_data = [], []
    plot_window_data = []
    base_metric_windows_labels = [
        "Jitter windows on original MDC (aiming2_subject3)",
    ]

    base_metric_labels = [
        "MDC with applied jitter (aiming2_subject3)",
    ]

    fft_metric_labels = [
        "MDCSS with applied jitter (aiming2_subject3)",
    ]

    ds_configs = [
        "data/datasets/publication_configs/lafan1_custom_jitter.json",
    ]

    highlight_frames = []

    # zoom_x_range = [
    #     (6850, 7150),
    #     (450, 750),
    # ]
    #
    # zoom_y_range_base_metric = [
    #     (0, 0.1),
    #     (0, 0.1),
    # ]
    #
    # zoom_y_range_fft_metric = [
    #     (0, 2),
    #     (0, 2),
    # ]

    y_range = (0, 35)

    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        for i, data in enumerate(dataset_analyzer.train_loader):
            (lpos, lrot, joint_offsets,
             foot_contact, parents, bone_lengths, sequence_name, data_idx) = data
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
                highlight_frames.append((window_noise_start, window_noise_end))

            _, new_y_data, _, label = calculate_metrics(dataset_analyzer, n=2, noise_windows=None)
            plot_window_data.append(new_y_data)
            _, noisy_y_data, noisy_y_std_data, noisy_label = calculate_metrics(
                dataset_analyzer,
                n=2,
                noise_windows=highlight_frames,
                # jitter_std=0.01,
            )
            plot_y_data.append(noisy_y_data)
            plot_y_std_data.append(noisy_y_std_data)

    for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
        for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
            jitter_window_title = base_metric_windows_labels[metric_idx]
            base_metric_title = base_metric_labels[metric_idx]
            fft_metric_title = fft_metric_labels[metric_idx]
            fft_data = plot_y_std_data[metric_idx][i]
            frames_to_highlight = highlight_frames[metric_idx]
            # x_range = zoom_x_range[metric_idx]
            # y_range = zoom_y_range_base_metric[metric_idx]
            # y_fft_range = zoom_y_range_fft_metric[metric_idx]

            window_y_data = plot_window_data[metric_idx][i].flatten().cpu().numpy()
            window_frame_range = range(len(window_y_data))
            window_plot_row = 0
            axis[window_plot_row].plot(window_frame_range, window_y_data)
            axis[window_plot_row].set_title(jitter_window_title, size=FONT_SIZE)
            axis[window_plot_row].set_xlabel("frame", fontsize=FONT_SIZE)
            axis[window_plot_row].set_ylabel("MDC", fontsize=FONT_SIZE)
            axis[window_plot_row].set_ylim(y_range)

            for start, end in highlight_frames:
                axis[window_plot_row].axvspan(start, end, color='b', alpha=0.25)

            metric_y_data = file_y_data.flatten().cpu().numpy()
            base_metric_frame_range = range(len(metric_y_data))
            regular_plot_row = 1
            axis[regular_plot_row].plot(base_metric_frame_range, metric_y_data)
            axis[regular_plot_row].set_title(base_metric_title, size=FONT_SIZE)
            axis[regular_plot_row].set_xlabel("frame", fontsize=FONT_SIZE)
            axis[regular_plot_row].set_ylabel("MDC", fontsize=FONT_SIZE)
            # axis[regular_plot_row, 0].axvspan(frames_to_highlight[0], frames_to_highlight[1],
            #                                                            color='b', alpha=0.25)
            # axis[regular_plot_row, 0].set_xlim(x_range)
            axis[regular_plot_row].set_ylim(y_range)

            fft_y_data = fft_data.flatten().cpu().numpy()
            fft_frame_range = range(len(fft_y_data))
            fft_plot_row = 2
            axis[fft_plot_row].plot(fft_frame_range, fft_y_data, color='g')
            axis[fft_plot_row].set_title(fft_metric_title, size=FONT_SIZE)
            axis[fft_plot_row].set_xlabel("window start frame", fontsize=FONT_SIZE)
            axis[fft_plot_row].set_ylabel("MDCSS", fontsize=FONT_SIZE)
            # axis[std_plot_row, 1].axvspan(frames_to_highlight[0], frames_to_highlight[1],
            #                                                        color='b', alpha=0.25)
            # axis[std_plot_row, 1].set_xlim(x_range)
            # axis[std_plot_row, 1].set_ylim(y_fft_range)

            warning_intervals = []
            error_intervals = []

            interval_start, interval_level = -1, 0
            for j, val in enumerate(fft_y_data):
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
                    if interval_level == 0:
                        interval_start = j
                    elif interval_level == 2:
                        interval_end = j
                        error_intervals.append((interval_start, interval_end))
                        interval_start = j
                    interval_level = 1
                if val > 20:
                    if interval_level == 0:
                        interval_start = j
                    elif interval_level == 1:
                        interval_end = j
                        warning_intervals.append((interval_start, interval_end))
                        interval_start = j
                    interval_level = 2

            # start, end = 616 // 4, 650 // 4
            for warning_info in warning_intervals:
                # print(f"Warning value at frames: {warning_info}")
                axis[fft_plot_row].axvspan(warning_info[0], warning_info[1], color='y', alpha=0.25)

            for error_info in error_intervals:
                # print(f"Error value at frames: {error_info}")
                # axis[regular_plot_row, 0].axvspan(start, end, color='red', alpha=0.25)
                axis[fft_plot_row].axvspan(error_info[0], error_info[1], color='r', alpha=0.5)

    plt.savefig("custom_jitter_0125.pdf")
