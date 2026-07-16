import mne
import numpy as np
import matplotlib.pyplot as plt

BANDS = {
    'Delta (1-4 Hz)': (1, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30)
}

def _compute_band_power(epochs, fmin, fmax):
    """Compute average band power across epochs for all channels in decibels (10 * log10)."""
    if len(epochs) == 0:
        return np.zeros(len(epochs.info['ch_names']))
    spectrum = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False)
    psd_mean = spectrum.get_data().mean(axis=0)
    band_power = psd_mean.mean(axis=1)
    return 10 * np.log10(band_power)

def plot_alert_vs_drowsy_topomaps(epochs_alert, epochs_drowsy, title, out_path, cmap='viridis', include_title=True):
    """
    Plots side-by-side topomaps of Alert vs Drowsy for different frequency bands.
    Ensures that the color scale (vmin, vmax) is identical for direct comparison.
    """

    epochs_alert.set_montage('standard_1020')
    epochs_drowsy.set_montage('standard_1020')
    info = epochs_alert.info

    fig, axes = plt.subplots(2, len(BANDS), figsize=(3 * len(BANDS), 6))
    if include_title:
        fig.suptitle(title, fontsize=16, y=1.05)

    for i, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
        # Compute band power
        power_alert = _compute_band_power(epochs_alert, fmin, fmax)
        power_drowsy = _compute_band_power(epochs_drowsy, fmin, fmax)

        # Determine common scale for this band
        vmin = min(power_alert.min(), power_drowsy.min())
        vmax = max(power_alert.max(), power_drowsy.max())

        # Plot alert
        ax_alert = axes[0, i]
        im_alert, _ = mne.viz.plot_topomap(
            power_alert, info, axes=ax_alert, show=False,
            cmap=cmap, vlim=(vmin, vmax), sphere='eeglab'
        )
        if i == 0:
            ax_alert.set_ylabel('Alert', fontsize=14, fontweight='bold', labelpad=20)
        ax_alert.set_title(band_name, fontsize=12)

        # Plot drowsy
        ax_drowsy = axes[1, i]
        im_drowsy, _ = mne.viz.plot_topomap(
            power_drowsy, info, axes=ax_drowsy, show=False,
            cmap=cmap, vlim=(vmin, vmax), sphere='eeglab'
        )
        if i == 0:
            ax_drowsy.set_ylabel('Drowsy', fontsize=14, fontweight='bold', labelpad=20)

        # Add colorbar per band
        cbar_ax = fig.add_axes([ax_alert.get_position().x1 + 0.01, 
                                ax_drowsy.get_position().y0, 
                                0.01, 
                                ax_alert.get_position().y1 - ax_drowsy.get_position().y0])
        plt.colorbar(im_alert, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close()
