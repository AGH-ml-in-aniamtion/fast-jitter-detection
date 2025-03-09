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
        "LAFAN1 example (aiming1_subject1)",
        "Contemporary Dance example (Sophie_Afraid-01)",
    ]

    fft_metric_labels = [
        "LAFAN1 example (aiming1_subject1)",
        "Contemporary Dance example (Sophie_Afraid-01)",
    ]

    ds_configs = [
        "data/datasets/publication_configs/lafan1_stomp_fft.json",
        "data/datasets/publication_configs/contemporary_jitter_fft.json",
    ]

    highlight_frames_mdc = [
        (6985, 6995),
        (605, 655),
    ]

    highlight_frames_mdcss = [
        (6975, 6995),
        (570, 655),
    ]

    zoom_x_range = [
        (6850, 7150),
        (450, 750),
    ]

    zoom_y_range_base_metric = [
        (0, 5),
        (0, 5),
    ]

    zoom_y_range_fft_metric = [
        (0, 15),
        (0, 15),
    ]

    font_size = 14
    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        _, new_y_data, y_std_data, label = calculate_metrics(dataset_analyzer)
        plot_y_data.append(new_y_data)
        plot_y_std_data.append(y_std_data)

    for metric_idx, y_axis_data_batches in enumerate(plot_y_data):
        for i, file_y_data in enumerate(y_axis_data_batches):  # Unused
            base_metric_title = base_metric_labels[metric_idx]
            fft_metric_title = fft_metric_labels[metric_idx]
            fft_data = plot_y_std_data[metric_idx][i]
            frames_to_highlight_mdc = highlight_frames_mdc[metric_idx]
            frames_to_highlight_mdcss = highlight_frames_mdcss[metric_idx]
            x_range = zoom_x_range[metric_idx]
            y_range = zoom_y_range_base_metric[metric_idx]
            y_fft_range = zoom_y_range_fft_metric[metric_idx]

            metric_y_data = file_y_data.flatten().cpu().numpy()
            base_metric_frame_range = range(len(metric_y_data))
            regular_plot_row = metric_idx
            axis[regular_plot_row, 0].plot(base_metric_frame_range, metric_y_data)
            axis[regular_plot_row, 0].set_title(base_metric_title, size=font_size)
            axis[regular_plot_row, 0].set_xlabel("frame", fontsize=font_size)
            axis[regular_plot_row, 0].set_ylabel("MDC", fontsize=font_size)
            axis[regular_plot_row, 0].axvspan(frames_to_highlight_mdc[0], frames_to_highlight_mdc[1],
                                              color='b', alpha=0.25)
            axis[regular_plot_row, 0].set_xlim(x_range)
            axis[regular_plot_row, 0].set_ylim(y_range)

            fft_y_data = fft_data.flatten().cpu().numpy()
            fft_frame_range = range(len(fft_y_data))
            std_plot_row = metric_idx
            axis[std_plot_row, 1].plot(fft_frame_range, fft_y_data, color='g')
            axis[std_plot_row, 1].set_title(fft_metric_title, size=font_size)
            axis[std_plot_row, 1].set_xlabel("frame", fontsize=font_size)
            axis[std_plot_row, 1].set_ylabel("MDCSS", fontsize=font_size)
            axis[std_plot_row, 1].axvspan(frames_to_highlight_mdcss[0], frames_to_highlight_mdcss[1],
                                          color='b', alpha=0.25)
            axis[std_plot_row, 1].set_xlim(x_range)
            axis[std_plot_row, 1].set_ylim(y_fft_range)

    plt.savefig("fft_comparison_lafan1_contemporary.pdf")
