import matplotlib
import matplotlib.pyplot as plt

from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

FONT_SIZE = 14
matplotlib.rcParams.update({'font.size': FONT_SIZE})

if __name__ == "__main__":
    figure, axis = plt.subplots(2, 3, constrained_layout=True, figsize=(16, 8))

    plot_y_data, plot_y_std_data = [], []
    base_metric_labels = [
        "Skeleton scale x1",
        "Skeleton scale x0.1",
        "Skeleton scale x2",
    ]

    fft_metric_labels = [
        "Skeleton scale x1",
        "Skeleton scale x0.1",
        "Skeleton scale x2",
    ]

    ds_configs = [
        "data/datasets/publication_configs/contemporary_scale1.json",
        "data/datasets/publication_configs/contemporary_scale01.json",
        "data/datasets/publication_configs/contemporary_scale2.json",
    ]

    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        _, new_y_data, y_std_data, label = calculate_metrics(dataset_analyzer, n=2)
        plot_y_data.append(new_y_data)
        plot_y_std_data.append(y_std_data)

    for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
        for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
            base_metric_title = base_metric_labels[metric_idx]
            fft_metric_title = fft_metric_labels[metric_idx]
            fft_data = plot_y_std_data[metric_idx][i]

            metric_y_data = file_y_data.flatten().cpu().numpy()
            base_metric_frame_range = range(len(metric_y_data))
            regular_plot_row = 0
            regular_plot_col = metric_idx
            axis[regular_plot_row, regular_plot_col].plot(base_metric_frame_range, metric_y_data)
            axis[regular_plot_row, regular_plot_col].set_title(base_metric_title, size=FONT_SIZE)
            axis[regular_plot_row, regular_plot_col].set_xlabel("frame", fontsize=FONT_SIZE)
            axis[regular_plot_row, regular_plot_col].set_ylabel("MDC", fontsize=FONT_SIZE)

            fft_y_data = fft_data.flatten().cpu().numpy()
            fft_frame_range = range(len(fft_y_data))
            fft_plot_row = 1
            fft_plot_col = metric_idx
            axis[fft_plot_row, fft_plot_col].plot(fft_frame_range, fft_y_data, color='g')
            axis[fft_plot_row, fft_plot_col].set_title(fft_metric_title, size=FONT_SIZE)
            axis[fft_plot_row, fft_plot_col].set_xlabel("frame", fontsize=FONT_SIZE)
            axis[fft_plot_row, fft_plot_col].set_ylabel("MDCSS", fontsize=FONT_SIZE)

            warning_intervals = []
            error_intervals = []

    plt.savefig("fft_scale_comparison_no_windows.pdf")
