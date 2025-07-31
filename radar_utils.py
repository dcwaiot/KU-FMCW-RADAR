import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from collections import Counter
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error



# def range_resolution(Start_frequency,
#                      End_frequncy,
#                      ramp_end_time_us,
#                      light_speed=3e8):
#     freq_slope_const_mhz_per_us = (End_frequncy - Start_frequency) *1e3/ ramp_end_time_us
#     print(freq_slope_const_mhz_per_us)
#     bandwidth_hz = freq_slope_const_mhz_per_us * ramp_end_time_us * 1e6
#     print(ramp_end_time_us)
#     range_res_m = light_speed / (2.0 * bandwidth_hz)
#     print(range_res_m)

#     return range_res_m, bandwidth_hz

def range_resolution(num_adc_samples, dig_out_sample_rate, Start_frequency, End_frequncy, ramp_end_time_us):
    light_speed = 3e8  # m/s


    freq_slope_mhz_per_us = (End_frequncy - Start_frequency)*1e3 / ramp_end_time_us
    # freq_slope_mhz_per_us = 99.987

    # ADC sampling period (microseconds)
    adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
    # adc_sample_period_usec = ramp_end_time_us
    # Bandwidth (Hz)
    bandwidth_hz = freq_slope_mhz_per_us * adc_sample_period_usec * 1e6

    # Range resolution (meters)
    range_res_m = light_speed / (2.0 * bandwidth_hz)

    return range_res_m, bandwidth_hz





def organize_1843(raw_frame, num_chirps, num_rx, adc_sample, frame_num):
    ret = np.zeros(len(raw_frame) // 2, dtype=complex)
    ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
    ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
    return ret.reshape(frame_num, num_chirps, num_rx, adc_sample)


def clutter_removal(input_val, axis=0):
    reordering = np.arange(len(input_val.shape))
    reordering[0], reordering[axis] = axis, 0
    input_val = input_val.transpose(reordering)
    mean = input_val.mean(axis=0)
    output_val = input_val - mean
    return output_val.transpose(np.argsort(reordering))


def plot_fft_surface_views(x, y, z_orig, z_scr):
    # Zoom into bins 10 to 19
    x = np.arange(0, 20)  # Bin indices
    # range_zoom = z_orig[:, 10:20]       # Shape: (frames, 10 bins)
    # range_zoom_scr = z_scr[:, 10:20]
    fig1 = plt.figure(figsize=(16, 12), num="FFT Surface Views")
    x_plot, y_plot = np.meshgrid(x, y)

    ax1 = fig1.add_subplot(221)
    cs1 = ax1.contourf(x, y, z_orig[:,0:20], cmap='viridis')
    fig1.colorbar(cs1, ax=ax1)
    ax1.set_title('Original FFT - 2D')

    ax2 = fig1.add_subplot(222, projection='3d')
    ax2.plot_surface(x_plot, y_plot, z_orig[:,0:20], cmap='viridis')
    ax2.set_title('Original FFT - 3D')

    ax3 = fig1.add_subplot(223)
    cs2 = ax3.contourf(x, y, z_scr[:,0:20], cmap='viridis')
    fig1.colorbar(cs2, ax=ax3)
    ax3.set_title('Mean-Subtracted FFT - 2D')

    ax4 = fig1.add_subplot(224, projection='3d')
    ax4.plot_surface(x_plot, y_plot, z_scr[:,0:20], cmap='viridis')
    ax4.set_title('Mean-Subtracted FFT - 3D')

    plt.tight_layout()
    plt.show()


from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def plot_max_bin_indices_per_txrx(fft_data_scr):
    fig2, axes = plt.subplots(1, 8, figsize=(32, 5), num="Max Bin Index Per TxRx (X=Bin, Y=Frame)")
    
    for txrx in range(8):
        ax = axes[txrx]
        # Get max bin index per frame for current TxRx
        max_bin_per_frame = [
            np.argmax(np.abs(fft_data_scr[f, 0, txrx, :]))
            for f in range(fft_data_scr.shape[0])
        ]

        bin_counts = Counter(max_bin_per_frame)
        most_common = bin_counts.most_common(5)

        # === Plot: X = Bin Index, Y = Frame Index ===
        ax.scatter(max_bin_per_frame, np.arange(len(max_bin_per_frame)), s=1.5, color='blue')

        # Annotate top 3 bins
        for i, (bin_idx, count) in enumerate(most_common):
            ax.text(0.5, 0.9 - i*0.06,
                    f'{i+1}st Bin: {bin_idx} (Count: {count})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='brown')

        ax.set_title(f'Chirp 0 - TxRx {txrx}', fontsize=10)
        ax.set_xlabel('Bin Index')
        if txrx == 0:
            ax.set_ylabel('Frame Index')
        ax.set_xlim(9, 16)

        ax.set_xticks(np.arange(9,16, 1))  

        ax.set_ylim(0, fft_data_scr.shape[0])
        ax.grid(True)

    plt.tight_layout()
    plt.show()





def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate steering vectors for AoA estimation."""
    num_vec = int(round(2 * ang_est_range / ang_est_resolution + 1))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex128')
    
    for kk in range(num_vec):
        angle_deg = -ang_est_range + kk * ang_est_resolution
        for jj in range(num_ant):
            phase = -1 * np.pi * jj * np.sin(np.radians(angle_deg))
            steering_vectors[kk, jj] = np.exp(1j * phase)
    
    return steering_vectors

def aoa_bartlett_batch(fft_data, bins_of_interest, ang_est_range=90, ang_est_resolution=1):

    num_frames = fft_data.shape[0]
    num_antennas = fft_data.shape[2]
    num_angles = int(round(2 * ang_est_range / ang_est_resolution + 1))
    angles = np.linspace(-ang_est_range, ang_est_range, num_angles)
    steering_vectors = gen_steering_vec(ang_est_range, ang_est_resolution, num_antennas)

    deg_array = {bin_idx: [] for bin_idx in bins_of_interest}
    den_array = {bin_idx: [] for bin_idx in bins_of_interest}

    for frame in range(num_frames):
        for bin_index in bins_of_interest:
            signal = fft_data[frame, :, :, bin_index].T  # shape: (antennas, chirps)
            y = np.matmul(np.conjugate(steering_vectors), signal)  # (angles, chirps)
            spectrum = np.sum(np.abs(y) ** 2, axis=1)              # (angles,)
            peak_angle = angles[np.argmax(spectrum)]

            deg_array[bin_index].append(peak_angle)
            den_array[bin_index].append(spectrum)

    # Convert to numpy arrays
    for bin_index in bins_of_interest:
        deg_array[bin_index] = np.array(deg_array[bin_index])    # (frames,)
        den_array[bin_index] = np.array(den_array[bin_index])    # (frames, angles)

    return deg_array, den_array


def plot_aoa_deg_comparison(deg_array_dict, bins=[11, 12, 13], frame_range=(0, None), colors=None):

    if colors is None:
        colors = ['r', 'g', 'b', 'm', 'c', 'y']

    plt.figure(figsize=(20, 5))
    start, end = frame_range
    for i, bin_idx in enumerate(bins):
        aoa_vals = deg_array_dict[bin_idx]
        if end is None:
            end = len(aoa_vals)
        plt.plot(np.arange(start, end), aoa_vals[start:end], label=f"Bin {bin_idx}", color=colors[i % len(colors)])

    plt.title("AoA Angle (deg) Comparison Across Bins")
    plt.xlabel("Frame Index")
    plt.ylabel("Estimated AoA Angle (degrees)")
    plt.xlim(start, end)
    plt.xticks(np.arange(start, end, 50))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def print_top_aoa_angles_multi(deg_array, bin_indices=[11, 12, 13], top_n=5):

    result = {}
    print(f"\nTop {top_n} Most Frequent AoA Angles per Bin:")
    for bin_idx in bin_indices:
        angle_counts = Counter(deg_array[bin_idx])
        top_angles = angle_counts.most_common(top_n)
        result[bin_idx] = top_angles

        print(f"\nBin {bin_idx}:")
        for angle, count in top_angles:
            print(f"  Angle: {angle:.1f}°, Count: {count}")

    return result



def plot_unwrapped_phase_over_bins(
    fft_data,
    chirp=0,
    rangebins=None,
    bin_txrx_pairs=None,
    frame_limit=None,
    step=None,
    fs=20,  # Sampling frequency (Hz) for filtering
    low_cut=None,
    high_cut=None
):
    num_frames = fft_data.shape[0]
    num_txrx = fft_data.shape[2]

    fig, axs = plt.subplots(7, 1, figsize=(25, 20))

    for i, rbin in enumerate(rangebins):
        for txrx in range(num_txrx):
            phase_rad = np.angle(fft_data[:, chirp, txrx, rbin])
            unwrapped_rad = np.unwrap(phase_rad)
            axs[i].plot(np.arange(num_frames), unwrapped_rad, label=f"TxRx {txrx}")
        
        axs[i].set_xlabel("Frame Index")
        axs[i].set_ylabel("Unwrapped Phase (rad)")
        axs[i].set_title(f"Unwrapped Phase - Chirp {chirp}, Range Bin {rbin}")
        axs[i].legend(loc="upper right")
        axs[i].set_xlim(0, frame_limit)
        axs[i].set_xticks(np.arange(0, frame_limit + 1, step))
        axs[i].grid(True)

    # Plot specified bin-TxRx pairs on last subplot
    for bin_idx, txrx in bin_txrx_pairs:
        phase_rad = np.angle(fft_data[:, chirp, txrx, bin_idx])
        unwrapped_rad = np.unwrap(phase_rad)
        axs[5].plot(np.arange(num_frames), unwrapped_rad, label=f"Bin {bin_idx}, TxRx {txrx}")
        
    axs[5].set_ylabel("Unwrapped Phase (rad)")
    axs[5].legend(loc="upper right")
    axs[5].set_xlim(0, frame_limit)
    axs[5].set_xticks(np.arange(0, frame_limit + 1, step))
    axs[5].set_title(f"Unwrapped Phase (Original) - Chirp {chirp}")
    axs[5].grid(True)

    # plt.tight_layout()
    # plt.show()

        # --- Subplot for filtered bin-txrx pairs ---
    for bin_idx, txrx in bin_txrx_pairs:
        phase_rad = np.angle(fft_data[:, chirp, txrx, bin_idx])
        unwrapped_rad = np.unwrap(phase_rad)
        filtered_rad = BPF_filtfilt(unwrapped_rad, order=4, low_cut=low_cut, high_cut=high_cut, fs=fs)

        axs[6].plot(np.arange(num_frames), filtered_rad, label=f"Bin {bin_idx}, TxRx {txrx} (filtered)")

    axs[6].set_ylabel("Filtered Unwrapped Phase (rad)")
    axs[6].legend(loc="upper right")
    axs[6].set_xlim(0, frame_limit)
    axs[6].set_xticks(np.arange(0, frame_limit + 1, step))
    axs[6].set_title(f"Filtered Phase - Chirp {chirp}")
    axs[6].grid(True)

    plt.tight_layout()
    plt.show()







import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def BPF_filtfilt(sig, order, low_cut, high_cut, fs):
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, sig)

def analyze_phase_fft_segments_sliding(
    fft_data,
    fs=None,
    frame_period=None,
    segment_length=None,
    desired_segments=None,
    start_frame=0,
    bins_txrx_pairs=None,
    low_cut=None,
    high_cut=None,
    rows_per_fig=None,
    beamformed=False,
    steering_vectors=None
):


    def BPF_filtfilt(sig, order, low_cut, high_cut, fs):
        nyquist = 0.5 * fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, sig)

    chirp = 0
    total_frames = fft_data.shape[0]
    max_start = total_frames - segment_length
    # step_size = max_start // (desired_segments - 1)
    step_size = 20
    segment_indices = range(start_frame, total_frames - segment_length + 1, step_size)
    num_segments = len(segment_indices)

    results = []

    for idx, radar_start in enumerate(segment_indices):
        radar_end = radar_start + segment_length

        for bin_idx, txrx in bins_txrx_pairs:
            if beamformed:
                assert steering_vectors is not None, "steering_vectors must be provided for beamformed mode"
                if bin_idx in [11]:
                    w = steering_vectors[91].conj()
                elif bin_idx == 13:
                    w = steering_vectors[89].conj()
                else:
                    w = steering_vectors[90].conj()
                x_raw = fft_data[:, chirp, :, bin_idx]  # shape: (frames, antennas)
                x = x_raw @ w
            else:
                x = fft_data[:, chirp, txrx, bin_idx]

            phase_rad = np.angle(x)
            unwrapped = np.unwrap(phase_rad)
            unwrapped_filtered = BPF_filtfilt(unwrapped, order=4, low_cut=low_cut, high_cut=high_cut, fs=fs)

            freqs_unwrapped = np.fft.rfftfreq(4 * segment_length, d=1 / fs)
            fft_unwrapped = np.abs(np.fft.rfft(unwrapped_filtered[radar_start:radar_end], n=4 * segment_length))
            peak_unwrapped_freq = freqs_unwrapped[np.argmax(fft_unwrapped)]

            phase_diff_filtered = np.diff(unwrapped_filtered[radar_start:radar_end]) / (2 * np.pi * frame_period)
            freqs_diff = np.fft.rfftfreq(4 * segment_length, d=1 / fs)
            fft_diff = np.abs(np.fft.rfft(phase_diff_filtered, n=4 * segment_length))
            peak_diff_freq = freqs_diff[np.argmax(fft_diff)]

            results.append({
                "segment": idx,
                "start_frame": radar_start,
                "bin": bin_idx,
                "txrx": txrx,
                "peak_unwrapped_freq": peak_unwrapped_freq,
                "peak_diff_freq": peak_diff_freq
            })

    # Plot in batches
    for fig_idx in range(0, num_segments, rows_per_fig):
        seg_subset = segment_indices[fig_idx:fig_idx + rows_per_fig]
        fig, axs = plt.subplots(len(seg_subset), 4, figsize=(30, 5 * len(seg_subset)))

        if len(seg_subset) == 1:
            axs = np.expand_dims(axs, axis=0)

        for idx, radar_start in enumerate(seg_subset):
            radar_end = radar_start + segment_length
            row_ax = axs[idx]

            for bin_idx, txrx in bins_txrx_pairs:
                if beamformed:
                    if bin_idx in [11]:
                        w = steering_vectors[91].conj()
                    elif bin_idx == 13:
                        w = steering_vectors[89].conj()
                    else:
                        w = steering_vectors[90].conj()
                    x = fft_data[:, 0, :, bin_idx] @ w
                else:
                    x = fft_data[:, 0, txrx, bin_idx]

                phase_rad = np.angle(x)
                unwrapped = np.unwrap(phase_rad)
                filtered = BPF_filtfilt(unwrapped, order=4, low_cut=low_cut, high_cut=high_cut, fs=fs)
                segment = filtered[radar_start:radar_end]
                diff = np.diff(segment) / (2 * np.pi * frame_period)

                freqs = np.fft.rfftfreq(4 * segment_length, d=1 / fs)
                fft_unwrapped = np.abs(np.fft.rfft(segment, n=4 * segment_length))
                fft_diff = np.abs(np.fft.rfft(diff, n=4 * segment_length))

                peak_unwrapped_freq = freqs[np.argmax(fft_unwrapped)]
                peak_diff_freq = freqs[np.argmax(fft_diff)]

                row_ax[0].plot(segment, label=f"Bin {bin_idx}, TxRx {txrx}")
                row_ax[1].plot(diff, label=f"Bin {bin_idx}, TxRx {txrx}")
                row_ax[2].plot(freqs, fft_unwrapped)
                row_ax[2].axvline(peak_unwrapped_freq, color='r', linestyle='--')
                row_ax[2].set_xlim(0, 2)
                row_ax[2].set_xticks(np.arange(0, 2.1, 0.2))
                row_ax[3].plot(freqs, fft_diff)
                row_ax[3].axvline(peak_diff_freq, color='r', linestyle='--')
                row_ax[3].set_xlim(0, 2)
                row_ax[3].set_xticks(np.arange(0, 2.1, 0.2))

                for ax in row_ax:
                    ax.grid(True)
                    ax.legend(loc='upper right')

            row_ax[0].set_ylabel(f"Seg {radar_start}-{radar_end}")

        axs[-1][0].set_xlabel("Frame Index")
        axs[-1][1].set_xlabel("Frame Index")
        axs[-1][2].set_xlabel("Frequency (Hz)")
        axs[-1][3].set_xlabel("Frequency (Hz)")

        plt.tight_layout()
        plt.show()

    df = pd.DataFrame(results)
    return df, results




# def compare_resp_predictions_to_groundtruth(results, all_HR, bin_ids=None):


#     # --- Convert results list to array ---
#     Hearting_array = np.array([
#         [r["bin"], r["peak_unwrapped_freq"], r["peak_diff_freq"]] 
#         for r in results
#     ])
#     # print("Shape:", Hearting_array.shape)
#     # print(Hearting_array)

#     # --- Initialize storage ---
#     predictions = {}
#     errors = {}

#     # --- Trim ground truth to usable range ---
#     ref_gt = all_HR[:len(all_HR)]

#     for i, bin_id in enumerate(bin_ids):
#         bin_rows = Hearting_array[Hearting_array[:, 0] == bin_id]

#         bin_pred = bin_rows[:, 2] * 60  # Convert Hz → bpm
#         min_len = min(len(bin_pred), len(ref_gt))
#         bin_pred = bin_pred[:min_len]
#         ref_trimmed = ref_gt[:min_len]

#         predictions[f'bin_{bin_id}'] = bin_pred
#         errors[f'bin_{bin_id}'] = np.abs(bin_pred - ref_trimmed)

#     # --- Plot ---
#     xtick_pos = np.arange(len(ref_trimmed))
#     xtick_labels = (xtick_pos + 1) * 5
#     colors = ['r', 'g', 'b','m']

#     plt.figure(figsize=(20, 5))

#     # Subplot 1: Predicted vs GT
#     plt.subplot(2, 1, 1)
#     plt.plot(ref_trimmed, marker='s', label="Ground Truth", color='black')
#     for i, bin_id in enumerate(bin_ids):
#         key = f'bin_{bin_id}'
#         if key in predictions:
#             plt.plot(predictions[key], marker='o', linestyle='--', label=f"Bin {bin_id}", color=colors[i])
#     plt.title("Peak Unwrapped Frequency vs Ground Truth")
#     plt.ylabel("Frequency (bpm)")
#     plt.xticks(xtick_pos, labels=xtick_labels)
#     plt.grid(True)
#     plt.legend()

#     # Subplot 2: Error
#     plt.subplot(2, 1, 2)
#     for i, bin_id in enumerate(bin_ids):
#         key = f'bin_{bin_id}'
#         if key in errors:
#             plt.plot(errors[key], marker='x', label=f"Error Bin {bin_id}", color=colors[i])
#     plt.title("Estimation Error vs Ground Truth")
#     plt.ylabel("Absolute Error (bpm)")
#     plt.xlabel("Time (sec)")
#     plt.xticks(xtick_pos, labels=xtick_labels)
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
#     # --- Evaluation Metrics ---
#     metrics = {}
#     for bin_id in bin_ids:
#         key = f'bin_{bin_id}'
#         if key in predictions:
#             pred = predictions[key]
#             ref = ref_trimmed[:len(pred)]  # Align length just in case
#             mae = mean_absolute_error(ref, pred)
#             rmse = np.sqrt(mean_squared_error(ref, pred))
#             metrics[key] = {'MAE': mae, 'RMSE': rmse}
#             print(f"{key}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

#     return {
#         'ref_gt': ref_trimmed,
#         'predictions': predictions,
#         'errors': errors,
#         'metrics': metrics
#     }


def compare_resp_predictions_to_groundtruth(results, all_HR, bin_ids=None):
    if bin_ids is None:
        raise ValueError("You must provide a list of bin_ids to compare.")

    # --- Convert results list to array ---
    Hearting_array = np.array([
        [r["bin"], r["peak_unwrapped_freq"], r["peak_diff_freq"]] 
        for r in results
    ])

    # --- Initialize ---
    predictions = {}
    errors = {}
    ref_gt = all_HR[:len(all_HR)]
    
    for bin_id in bin_ids:
        bin_rows = Hearting_array[Hearting_array[:, 0] == bin_id]
        bin_pred = bin_rows[:, 2] * 60  # Hz → bpm
        print(bin_rows[:, 2].shape)
        min_len = min(len(bin_pred), len(ref_gt))
        bin_pred = bin_pred[:min_len]
        ref_trimmed = ref_gt[:min_len]

        predictions[f'bin_{bin_id}'] = bin_pred
        errors[f'bin_{bin_id}'] = np.abs(bin_pred - ref_trimmed)

    # --- Metrics for Ground Truth ---
    metrics = {}
    for bin_id in bin_ids:
        key = f'bin_{bin_id}'
        if key in predictions:
            pred = predictions[key]
            ref = ref_trimmed[:len(pred)]
            mae = mean_absolute_error(ref, pred)
            rmse = np.sqrt(mean_squared_error(ref, pred))
            metrics[key] = {'MAE': mae, 'RMSE': rmse}
            print(f"{key}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    # --- Compare Bin-to-Bin (relative performance) ---
    bin_metrics = {}
    reference_bin = bin_ids[0]
    ref_key = f'bin_{reference_bin}'

    print("\n--- Bin-to-Bin Comparisons ---")
    for bin_id in bin_ids:
        key = f'bin_{bin_id}'
        if key != ref_key:
            pred1 = predictions[key]
            pred2 = predictions[ref_key]
            min_len = min(len(pred1), len(pred2))
            mae_diff = mean_absolute_error(pred2[:min_len], pred1[:min_len])
            rmse_diff = np.sqrt(mean_squared_error(pred2[:min_len], pred1[:min_len]))
            bin_metrics[key] = {'MAE_vs_refbin': mae_diff, 'RMSE_vs_refbin': rmse_diff}
            print(f"{key} vs {ref_key}: MAE = {mae_diff:.2f}, RMSE = {rmse_diff:.2f}")

    # --- Plotting ---
    xtick_pos = np.arange(len(ref_trimmed))
    print(xtick_pos)
    xtick_labels = (xtick_pos + 1) 
    colors = ['r', 'g', 'b', 'm', 'c']

    plt.figure(figsize=(20, 5))

    # Subplot 1: Prediction vs GT
    plt.subplot(2, 1, 1)
    plt.plot(ref_trimmed, marker='s', label="Ground Truth", color='black')
    for i, bin_id in enumerate(bin_ids):
        key = f'bin_{bin_id}'
        if key in predictions:
            plt.plot(predictions[key], marker='o', linestyle='--', label=f"Bin {bin_id}", color=colors[i])
    plt.title("Predicted Frequency vs Ground Truth")
    plt.ylabel("Frequency (bpm)")
    # plt.yticks(np.arange(0, 31, 2))
    plt.yticks(np.arange(60, 110, 2))
    plt.xticks(xtick_pos, labels=xtick_labels)
    plt.grid(True)
    plt.legend()

    # Subplot 2: Absolute Error
    plt.subplot(2, 1, 2)
    for i, bin_id in enumerate(bin_ids):
        key = f'bin_{bin_id}'
        if key in errors:
            plt.plot(errors[key], marker='x', label=f"Error Bin {bin_id}", color=colors[i])
    plt.title("Estimation Error vs Ground Truth")
    plt.ylabel("Absolute Error (bpm)")
    plt.xlabel("Time (sec)")
    plt.xticks(xtick_pos, labels=xtick_labels)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'ref_gt': ref_trimmed,
        'predictions': predictions,
        'errors': errors,
        'metrics': metrics,
        'bin_vs_bin_metrics': bin_metrics
    }




def plot_txrx_snr_per_bin(fft_data, bins, chirp_idx=0, snr_range=(-5, 5)):
    statistics = {}
    num_txrx = fft_data.shape[2]
    fig = plt.figure(figsize=(30, 10))

    for bin_idx in bins:
        for txrx in range(num_txrx):
            signal = fft_data[:, chirp_idx, txrx, bin_idx]
            magnitude = np.abs(signal)

            # Estimate noise power using a moving average window
            noise_power = np.array([
                np.mean(magnitude[max(0, i - 2): min(len(magnitude), i + 3)])
                for i in range(len(magnitude))
            ])

            snr = 10 * np.log10(np.maximum(magnitude, 1e-12) / np.maximum(noise_power, 1e-12))

            stats = {
                'mean_snr': np.mean(np.abs(snr)),
                'min_snr': np.min(snr),
                'max_snr': np.max(snr),
                'peak_snr': np.max(snr),
                'variance': np.var(snr),
                'std': np.std(snr),
                'cv': np.std(snr) / (np.mean(np.abs(snr)) + 1e-6)
            }
            statistics[(bin_idx, txrx)] = stats  
            ax = fig.add_subplot(len(bins), num_txrx, (bin_idx - bins[0]) * num_txrx + txrx + 1)
            ax.plot(snr, label=f'Bin {bin_idx}, TxRx {txrx}')
            ax.set_title(f'Bin {bin_idx}, TxRx {txrx}')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('SNR (dB)')
            ax.set_ylim(snr_range)
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()
    return statistics  






def plot_iq_3d_trajectory(fft_data, bins_txrx_pairs, chirp_idx=0, limit=None, title_prefix="SCR, DC removed IQ"):

    num_plots = len(bins_txrx_pairs)
    fig = plt.figure(figsize=(5 * num_plots, 5))

    for i, (bin_idx, txrx_idx) in enumerate(bins_txrx_pairs):
        x = fft_data[:, chirp_idx, txrx_idx, bin_idx].real
        y = fft_data[:, chirp_idx, txrx_idx, bin_idx].imag
        z = np.arange(len(x))

        x -= np.mean(x)
        y -= np.mean(y)

        ax = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        ax.scatter3D(x, y, z, c=z, cmap='viridis') 
        ax.plot(x, y, z, color='g', linewidth=2)
        # ax.plot(x, y, z, color='g')


        ax.set_title(f'{title_prefix} - Bin {bin_idx}, TxRx {txrx_idx}', fontsize=10)

    plt.tight_layout()
    plt.show()






def plot_bartlett_spectrum_with_peaks(
    spectrum_array_bartlett,
    bins_to_process,
    start_frame=0,
    end_frame=None,
    angle_range=(50, 140),
    title="Bartlett Spectrum per Bin"
):
    num_angles = list(spectrum_array_bartlett.values())[0].shape[1]
    angles = np.linspace(0, 180, num_angles)
    idx_range = np.where((angles >= angle_range[0]) & (angles <= angle_range[1]))[0]

    if end_frame is None:
        end_frame = list(spectrum_array_bartlett.values())[0].shape[0]

    peaks_bartlett = {bin_num: [] for bin_num in bins_to_process}
    
    fig, axes = plt.subplots(1, len(bins_to_process), figsize=(8 * len(bins_to_process), 6), sharey=True)

    if len(bins_to_process) == 1:
        axes = [axes]

    for i, bin_num in enumerate(bins_to_process):
        ax = axes[i]
        spectrum_bin = spectrum_array_bartlett[bin_num]
        angle_counter = Counter()

        for frame in range(start_frame, end_frame):
            spectrum_frame = spectrum_bin[frame]
            spectrum_range = spectrum_frame[idx_range]
            max_val = np.max(spectrum_range)
            max_angle = angles[idx_range[np.argmax(spectrum_range)]]

            peaks_bartlett[bin_num].append((frame, max_angle, max_val))
            angle_counter[round(max_angle, 1)] += 1

            ax.plot(angles, spectrum_frame, alpha=0.4)
            ax.plot(max_angle, max_val, 'ro')

        # Annotate top 5 angles
        top_angles = angle_counter.most_common(5)
        for j, (angle, count) in enumerate(top_angles):
            ax.text(0.5, 0.9 - j * 0.06,
                    f"{j+1}st Peak: {angle}° ({count})",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="brown")

        ax.set_title(f'Bin {bin_num}')
        ax.set_xlabel('Angle (degrees)')
        ax.set_xlim(angle_range)
        ax.set_xticks(np.arange(angle_range[0], angle_range[1]+1, 5))
        ax.grid(True)

    axes[0].set_ylabel("Spectrum Magnitude")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return peaks_bartlett

