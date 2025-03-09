import matplotlib
import matplotlib.pyplot as plt

from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

FONT_SIZE = 13
matplotlib.rcParams.update({'font.size': FONT_SIZE})

if __name__ == "__main__":
    use_window = [False, True]
    plots_per_col = len(use_window)

    base_metric_labels = [
        "LAFAN1 aiming",
        # "LAFAN1 dance2_subject4",
        "LAFAN1 fight",
        # "LAFAN1 obstacles",
        # "LAFAN1 obstacles fixed",
        "Contemporary",
    ]

    fft_metric_labels = [
        "LAFAN1 aiming",
        # "LAFAN1 dance2_subject4",
        "LAFAN1 fight",
        # "LAFAN1 obstacles",
        # "LAFAN1 obstacles fixed",
        "Contemporary",
    ]

    ds_configs = [
        "data/datasets/publication_configs/ablation_lafan1_aiming.json",
        # "data/datasets/publication_configs/ablation_lafan1_dance.json",
        "data/datasets/publication_configs/ablation_lafan1_fight.json",
        # "data/datasets/publication_configs/ablation_lafan1_obstacles.json",
        # "data/datasets/publication_configs/ablation_lafan1_obstacles_fixed.json",
        "data/datasets/publication_configs/ablation_contemporary_sophie.json",
    ]
    figure, axis = plt.subplots(len(use_window), len(ds_configs),
                                constrained_layout=True, figsize=(16, 8))

    y_ranges = [(0, 15), (0, 40), (0, 350)]

    for k, use_hamm_window in enumerate(use_window):
        plot_y_data, plot_y_std_data = [], []
        for ds_conf in ds_configs:
            dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
            # subsample_clip(dataset_analyzer, 4, "data/datasets/bvh/Sophie_Afraid-01_30fps.bvh")
            _, new_y_data, y_std_data, label = calculate_metrics(dataset_analyzer, n=2, use_hamm_window=use_hamm_window)
            # plot_x_data = new_x_data
            plot_y_data.append(new_y_data)
            plot_y_std_data.append(y_std_data)
            # metric_labels.append(label)

        for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
            for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
                base_metric_title = base_metric_labels[metric_idx]
                fft_metric_title = fft_metric_labels[metric_idx]
                fft_data = plot_y_std_data[metric_idx][i]

                # metric_y_data = file_y_data.flatten().cpu().numpy()
                # base_metric_frame_range = range(len(metric_y_data))
                # regular_plot_row = metric_idx
                # regular_plot_col = plots_per_row * k
                # axis[regular_plot_row, regular_plot_col].plot(base_metric_frame_range, metric_y_data)
                # axis[regular_plot_row, regular_plot_col].set_title(f"{base_metric_title} (window: {round(1/fps_div, 2)}s)", size=FONT_SIZE)
                # axis[regular_plot_row, regular_plot_col].set_xlabel("frame", fontsize=FONT_SIZE)
                # axis[regular_plot_row, regular_plot_col].set_ylabel("base metric value", fontsize=FONT_SIZE)
                # axis[regular_plot_row, regular_plot_col].set_ylim(y_range)

                fft_y_data = fft_data.flatten().cpu().numpy()
                fft_frame_range = range(len(fft_y_data))
                std_plot_row = k
                std_plot_col = metric_idx
                axis[std_plot_row, std_plot_col].plot(fft_frame_range, fft_y_data, color='g')
                axis[std_plot_row, std_plot_col].set_title(
                    f"{fft_metric_title} ({'rectangular' if not use_hamm_window else 'hamming'})", size=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_xlabel("frame", fontsize=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_ylabel("max from AC components", fontsize=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_ylim(y_ranges[metric_idx])

                # warning_intervals = []
                # error_intervals = []
                #
                # interval_start, interval_level = -1, 0
                # for j, val in enumerate(fft_y_data):
                #     if val < 10:
                #         if interval_level == 1:
                #             interval_end = j
                #             warning_intervals.append((interval_start, interval_end))
                #             interval_start = j
                #         elif interval_level == 2:
                #             interval_end = j
                #             error_intervals.append((interval_start, interval_end))
                #             interval_start = j
                #         interval_level = 0
                #     if 10 <= val < 25:
                #         if interval_level == 0:
                #             interval_start = j
                #         elif interval_level == 2:
                #             interval_end = j
                #             error_intervals.append((interval_start, interval_end))
                #             interval_start = j
                #         interval_level = 1
                #     if val >= 25:
                #         if interval_level == 0:
                #             interval_start = j
                #         elif interval_level == 1:
                #             interval_end = j
                #             warning_intervals.append((interval_start, interval_end))
                #             interval_start = j
                #         interval_level = 2
                #
                # for warning_info in warning_intervals:
                #     print(f"Warning value at frames: {warning_info}")
                #     axis[std_plot_row, std_plot_col].axvspan(warning_info[0], warning_info[1], color='y', alpha=0.25)
                #
                # for error_info in error_intervals:
                #     print(f"Error value at frames: {error_info}")
                #     # axis[regular_plot_row, 0].axvspan(start, end, color='red', alpha=0.25)
                #     axis[std_plot_row, std_plot_col].axvspan(error_info[0], error_info[1], color='r', alpha=0.5)

    # plt.tight_layout()
    plt.savefig("fft_except_DC.pdf")
