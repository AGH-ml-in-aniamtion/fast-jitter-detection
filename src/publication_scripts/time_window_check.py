import matplotlib
import matplotlib.pyplot as plt

from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

FONT_SIZE = 16
matplotlib.rcParams.update({'font.size': FONT_SIZE})

if __name__ == "__main__":
    fps_divisors = [10, 5, 4, 3, 2, 1.5, 1]
    plots_per_row = len(fps_divisors)

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
    figure, axis = plt.subplots(len(fps_divisors), len(ds_configs),
                                constrained_layout=True, figsize=(10, 16))

    y_ranges = [(0, 9), (0, 24), (0, 220)]
    y_ticks = [
        range(0, 9, 3),
        range(0, 25, 10),
        range(0, 250, 100),
    ]

    for k, fps_div in enumerate(fps_divisors):
        plot_y_data, plot_y_std_data = [], []
        for ds_conf in ds_configs:
            dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
            _, new_y_data, y_std_data, label = calculate_metrics(dataset_analyzer, n=2, fps_div=fps_div)
            plot_y_data.append(new_y_data)
            plot_y_std_data.append(y_std_data)

        for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
            for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
                base_metric_title = base_metric_labels[metric_idx]
                fft_metric_title = fft_metric_labels[metric_idx]
                fft_data = plot_y_std_data[metric_idx][i]

                fft_y_data = fft_data.flatten().cpu().numpy()
                fft_frame_range = range(len(fft_y_data))
                std_plot_row = k
                std_plot_col = metric_idx
                axis[std_plot_row, std_plot_col].plot(fft_frame_range, fft_y_data, color='g')
                axis[std_plot_row, std_plot_col].set_title(f"{fft_metric_title} (T={round(1 / fps_div, 2)}s)",
                                                           size=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_xlabel("frame", fontsize=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_ylabel("MDCSS", fontsize=FONT_SIZE)
                axis[std_plot_row, std_plot_col].set_ylim(y_ranges[metric_idx])
                axis[std_plot_row, std_plot_col].set_yticks(y_ticks[metric_idx])

    plt.savefig("time_windows.pdf")
