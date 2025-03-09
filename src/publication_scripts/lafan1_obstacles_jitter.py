import matplotlib
import matplotlib.pyplot as plt

from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

matplotlib.rcParams.update({'font.size': 14})

if __name__ == "__main__":
    plots_per_row = 2
    figure, axis = plt.subplots(2, plots_per_row, constrained_layout=True, figsize=(12, 8))

    plot_y_data, plot_y_std_data = [], []
    base_metric_labels = [
        # "Debug example",
        "LAFAN1 obstacles5_subject3 (original)",
        "LAFAN1 obstacles5_subject3 (manually fixed)",
        # "Contemporary Dance example (Sophie_Afraid-01)",
    ]

    fft_metric_labels = [
        "LAFAN1 obstacles5_subject3 (original)",
        "LAFAN1 obstacles5_subject3 (manually fixed)",
        # "Debug example",
        # "Contemporary Dance example (Sophie_Afraid-01)",
    ]

    ds_configs = [
        "data/datasets/publication_configs/lafan1_debug2.json",
        "data/datasets/publication_configs/lafan1_debug.json",
        # "data/datasets/publication_configs/contemporary_jitter_fft.json",
    ]

    font_size = 14
    y_range_metric = (0, 40)
    y_range_fft = (0, 80)

    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        # subsample_clip(dataset_analyzer, 4, "data/datasets/bvh/Sophie_Afraid-01_30fps.bvh")
        _, new_y_data, y_std_data, label = calculate_metrics(dataset_analyzer)
        # plot_x_data = new_x_data
        plot_y_data.append(new_y_data)
        plot_y_std_data.append(y_std_data)
        # metric_labels.append(label)

    for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
        for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
            base_metric_title = base_metric_labels[metric_idx]
            fft_metric_title = fft_metric_labels[metric_idx]
            fft_data = plot_y_std_data[metric_idx][i]

            metric_y_data = file_y_data.flatten().cpu().numpy()
            base_metric_frame_range = range(len(metric_y_data))
            regular_plot_row = metric_idx
            axis[regular_plot_row, 0].plot(base_metric_frame_range, metric_y_data)
            axis[regular_plot_row, 0].set_title(base_metric_title, size=font_size)
            axis[regular_plot_row, 0].set_xlabel("frame", fontsize=font_size)
            axis[regular_plot_row, 0].set_ylabel("MDC", fontsize=font_size)
            axis[regular_plot_row, 0].set_ylim(y_range_metric)

            fft_y_data = fft_data.flatten().cpu().numpy()
            fft_frame_range = range(len(fft_y_data))
            std_plot_row = metric_idx
            axis[std_plot_row, 1].plot(fft_frame_range, fft_y_data, color='g')
            axis[std_plot_row, 1].set_title(fft_metric_title, size=font_size)
            axis[std_plot_row, 1].set_xlabel("frame", fontsize=font_size)
            axis[std_plot_row, 1].set_ylabel("MDCSS", fontsize=font_size)
            axis[std_plot_row, 1].set_ylim(y_range_fft)

            dynamic_intervals = []
            warning_intervals = []
            error_intervals = []

            interval_start, interval_level = -1, 0
            for j, val in enumerate(fft_y_data):
                # Regular
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
                # Warning
                if 8 <= val <= 20:
                    if interval_level == 0:
                        interval_start = j
                    elif interval_level == 2:
                        interval_end = j
                        error_intervals.append((interval_start, interval_end))
                        interval_start = j
                    interval_level = 1
                # Error
                if val > 20:
                    if interval_level == 0:
                        interval_start = j
                    elif interval_level == 1:
                        interval_end = j
                        warning_intervals.append((interval_start, interval_end))
                        interval_start = j
                    interval_level = 2

            for warning_info in warning_intervals:
                print(f"Warning value at frames: {warning_info}")
                axis[std_plot_row, 1].axvspan(warning_info[0], warning_info[1], color='y', alpha=0.25)

            for error_info in error_intervals:
                print(f"Error value at frames: {error_info}")
                # axis[regular_plot_row, 0].axvspan(start, end, color='red', alpha=0.25)
                axis[std_plot_row, 1].axvspan(error_info[0], error_info[1], color='r', alpha=0.5)

    plt.savefig("lafan1_obstacles_jitter.pdf")
