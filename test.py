import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from filters import hampel_filter, butterworth_filter, resample_uniform


def visualize(nparr):
    if np.iscomplexobj(nparr):
        phase_data = np.unwrap(np.angle(nparr), axis=0)
    else:
        phase_data = nparr

    num_packets, num_subcarriers = phase_data.shape
    time_axis = np.arange(num_packets)

    plt.figure(figsize=(12, 6))

    for sc in range(num_subcarriers):
        plt.plot(time_axis, phase_data[:, sc], color="gray", alpha=0.3, linewidth=0.8)

    mean_waveform = np.mean(phase_data, axis=1)
    plt.plot(
        time_axis,
        mean_waveform,
        color="crimson",
        linewidth=2.5,
        label="Mean Respiration Waveform",
    )

    # Formatting elements to ensure the plot renders correctly
    plt.title(
        "WiFi CSI Raw Phase Waveform Across Subcarriers", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Packet Index", fontsize=12)
    plt.ylabel("Phase (radians)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.show()


def plot_spectrum(waveform, fs=100.0, band=(0.1, 0.5)):
    freqs, power = welch(waveform, fs=fs, nperseg=min(256, len(waveform)))

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, power)
    plt.axvspan(band[0], band[1], color="crimson", alpha=0.15, label="breathing band")

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if band_mask.any():
        peak_freq = freqs[band_mask][np.argmax(power[band_mask])]
        plt.axvline(peak_freq, color="crimson", linestyle="--")
        print(
            f"peak freq in band: {peak_freq:.3f} Hz ({peak_freq * 60:.1f} breaths/min)"
        )

    plt.xlim(0, 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum of Mean Waveform")
    plt.legend()
    plt.tight_layout()
    plt.show()


def count_breaths(waveform, fs, min_interval_sec=1.0):
    peaks, _ = find_peaks(waveform, distance=fs * min_interval_sec)
    duration_sec = len(waveform) / fs
    bpm = len(peaks) / duration_sec * 60
    return len(peaks), bpm


if __name__ == "__main__":
    df = pd.read_csv("data/20260905_143159_10.csv")
    print("df = ", df.shape)

    npy = np.load(df["data"].iloc[0])
    print("npy = ", npy.shape)

    # visualize(npy)
    # visualize(np.abs(npy))

    amp = np.abs(npy)
    timestamps = df["local_timestamp"].to_numpy()

    amp, time_axis, fs = resample_uniform(amp, timestamps)
    print(f"resampled: {amp.shape[0]} samples @ fs={fs:.2f} Hz")

    amp_df = pd.DataFrame(amp)
    cleaned_df = amp_df.apply(lambda col: hampel_filter(col)[0])
    cleaned_df = cleaned_df.apply(lambda col: butterworth_filter(col, fs=fs))

    visualize(cleaned_df.to_numpy())

    mean_waveform = cleaned_df.to_numpy().mean(axis=1)
    n_peaks, bpm = count_breaths(mean_waveform, fs)
    print(f"peak count: {n_peaks} -> {bpm:.1f} breaths/min")
