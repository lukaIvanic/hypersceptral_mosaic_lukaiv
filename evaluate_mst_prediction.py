import os
import glob
import h5py
import scipy.io
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# Settings
BANDS_COUNT = 31
WAVELENGTH_START = 400
WAVELENGTH_STEP = 10
WAVELENGTHS = [WAVELENGTH_START + i * WAVELENGTH_STEP for i in range(BANDS_COUNT)]

def compute_mrae(y_true, y_pred):
    """Mean Relative Absolute Error."""
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8))

def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_rse(y_true, y_pred):
    """Relative Squared Error (Global)."""
    return np.sum((y_true - y_pred) ** 2) / (np.sum(y_true ** 2) + 1e-8)

def compute_psnr(y_true, y_pred):
    """PSNR."""
    return psnr_func(y_true, y_pred, data_range=1.0)

def compute_ssim(y_true, y_pred):
    """SSIM."""
    return ssim_func(y_true, y_pred, data_range=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='data/track1/test-public/hsi_61')
    parser.add_argument('--pred_dir', type=str, default='../MST-plus-plus/predict_code/exp/mst_plus_plus_mosaic')
    parser.add_argument('--bands', type=int, default=31, help="Number of bands to evaluate (starting from 0)")
    parser.add_argument('--plot_output', type=str, default='evaluation_plots.png', help="Output file for plots")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.h5')))
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.mat')))

    if not gt_files:
        print(f"No GT files found in {args.gt_dir}")
        return
    if not pred_files:
        print(f"No prediction files found in {args.pred_dir}")
        return

    gt_dict = {os.path.basename(f).replace('.h5', ''): f for f in gt_files}
    pred_dict = {os.path.basename(f).replace('.mat', ''): f for f in pred_files}

    common_names = sorted(list(set(gt_dict.keys()) & set(pred_dict.keys())))
    print(f"Found {len(common_names)} common files to evaluate.")

    if len(common_names) == 0:
        return

    # Aggregated metrics (scalar)
    agg_metrics = {'mrae': [], 'rmse': [], 'rse': [], 'psnr': [], 'ssim': []}
    
    # Per-band metrics (list of arrays, one per image)
    # We'll average these over all images for the final plot
    band_metrics = {
        'mrae': np.zeros((len(common_names), args.bands)),
        'rmse': np.zeros((len(common_names), args.bands)),
        'psnr': np.zeros((len(common_names), args.bands)),
        'ssim': np.zeros((len(common_names), args.bands))
    }

    for idx, name in enumerate(common_names):
        gt_path = gt_dict[name]
        pred_path = pred_dict[name]
        
        print(f"Evaluating {name}...")

        # Load GT
        with h5py.File(gt_path, 'r') as f:
            gt = f['cube'][:]
        
        # Slice GT to desired bands (0 to 30)
        gt = gt[:, :, :args.bands]

        # Load Pred
        # Use scipy.io.loadmat for .mat files (Bug 1 fix)
        try:
            mat = scipy.io.loadmat(pred_path)
            pred = mat['cube']
        except NotImplementedError:
            # Fallback for v7.3 mat files which need h5py
            with h5py.File(pred_path, 'r') as f:
                pred = f['cube'][:]
        
        # Slice prediction to match the requested number of bands
        pred = pred[:, :, :args.bands]
        
        # Handle potential shape mismatch / transpose issues
        if gt.shape != pred.shape:
            if pred.shape == (gt.shape[2], gt.shape[1], gt.shape[0]):
                 pred = pred.transpose(2, 1, 0)
            elif pred.shape == (gt.shape[2], gt.shape[0], gt.shape[1]):
                 pred = pred.transpose(1, 2, 0)
            
            if gt.shape != pred.shape:
                print(f"Shape mismatch for {name}: GT {gt.shape} vs Pred {pred.shape}. Skipping.")
                continue
            
        # Global Metrics
        m_mrae = compute_mrae(gt, pred)
        m_rmse = compute_rmse(gt, pred)
        m_rse = compute_rse(gt, pred)
        m_psnr = compute_psnr(gt, pred)
        
        # SSIM (multichannel)
        # skimage ssim with channel_axis is convenient
        try:
            m_ssim = ssim_func(gt, pred, data_range=1.0, channel_axis=-1)
        except TypeError:
            m_ssim = ssim_func(gt, pred, data_range=1.0, multichannel=True)

        agg_metrics['mrae'].append(m_mrae)
        agg_metrics['rmse'].append(m_rmse)
        agg_metrics['rse'].append(m_rse)
        agg_metrics['psnr'].append(m_psnr)
        agg_metrics['ssim'].append(m_ssim)

        print(f"  MRAE: {m_mrae:.4f} | RSE: {m_rse:.4f} | RMSE: {m_rmse:.4f} | PSNR: {m_psnr:.2f} | SSIM: {m_ssim:.4f}")

        # Per-band Metrics
        for b in range(args.bands):
            gt_band = gt[:, :, b]
            pred_band = pred[:, :, b]
            
            band_metrics['mrae'][idx, b] = compute_mrae(gt_band, pred_band)
            band_metrics['rmse'][idx, b] = compute_rmse(gt_band, pred_band)
            band_metrics['psnr'][idx, b] = compute_psnr(gt_band, pred_band)
            band_metrics['ssim'][idx, b] = compute_ssim(gt_band, pred_band)

    if len(agg_metrics['mrae']) > 0:
        print("-" * 40)
        print(f"Average Metrics over {len(agg_metrics['mrae'])} samples:")
        print(f"MRAE: {np.mean(agg_metrics['mrae']):.4f}")
        print(f"RSE:  {np.mean(agg_metrics['rse']):.4f}")
        print(f"RMSE: {np.mean(agg_metrics['rmse']):.4f}")
        print(f"PSNR: {np.mean(agg_metrics['psnr']):.2f}")
        print(f"SSIM: {np.mean(agg_metrics['ssim']):.4f}")
        
        # Plotting
        avg_band_metrics = {
            'mrae': np.mean(band_metrics['mrae'], axis=0),
            'rmse': np.mean(band_metrics['rmse'], axis=0),
            'psnr': np.mean(band_metrics['psnr'], axis=0),
            'ssim': np.mean(band_metrics['ssim'], axis=0)
        }
        
        plot_metrics(avg_band_metrics, args.plot_output)
        print(f"Plots saved to {args.plot_output}")
    else:
        print("No samples evaluated.")

def plot_metrics(metrics_dict, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # MRAE
    axs[0, 0].plot(WAVELENGTHS, metrics_dict['mrae'], marker='o', color='r')
    axs[0, 0].set_title('MRAE per Band')
    axs[0, 0].set_xlabel('Wavelength (nm)')
    axs[0, 0].set_ylabel('MRAE')
    axs[0, 0].grid(True)

    # RSE (using RMSE here as per band proxy or just RMSE)
    # User asked for RSE, but RSE is usually a global metric. 
    # Plotting RMSE per band is standard.
    # If "Relative Squared Error" per band is needed: sum((y-x)^2)/sum(y^2) per band.
    # But here we stored RMSE. Let's plot RMSE.
    axs[0, 1].plot(WAVELENGTHS, metrics_dict['rmse'], marker='o', color='g')
    axs[0, 1].set_title('RMSE per Band')
    axs[0, 1].set_xlabel('Wavelength (nm)')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].grid(True)

    # PSNR
    axs[1, 0].plot(WAVELENGTHS, metrics_dict['psnr'], marker='o', color='b')
    axs[1, 0].set_title('PSNR per Band')
    axs[1, 0].set_xlabel('Wavelength (nm)')
    axs[1, 0].set_ylabel('PSNR (dB)')
    axs[1, 0].grid(True)

    # SSIM
    axs[1, 1].plot(WAVELENGTHS, metrics_dict['ssim'], marker='o', color='m')
    axs[1, 1].set_title('SSIM per Band')
    axs[1, 1].set_xlabel('Wavelength (nm)')
    axs[1, 1].set_ylabel('SSIM')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.close()

if __name__ == '__main__':
    main()

