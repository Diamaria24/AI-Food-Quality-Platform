from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral import envi
from scipy.stats import spearmanr

PROJECT_ROOT = Path(r'D:\Projects\AI-Food-Quality-Platform')
RAW_DIR = PROJECT_ROOT / 'datasets' / 'raw' / 'hyperspectral' / 'pistachio_hsi'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'pistachio_spectral_signatures'
RANDOM_STATE = 42
SPECTRA_PER_CUBE = 2000
REFLECTANCE_SCALE = 10000.0
AFB1_THRESHOLD = 8.0
AFB1_LEVELS = {
    1: 0.00, 2: 0.40, 3: 0.67, 4: 0.88, 5: 1.13, 6: 1.66,
    7: 2.15, 8: 2.30, 9: 2.48, 10: 2.82, 11: 3.01, 12: 3.05,
    13: 3.85, 14: 4.43, 15: 5.12, 16: 5.30, 17: 6.37,
    18: 8.93, 19: 12.16, 20: 17.12, 21: 24.03, 22: 26.14,
    23: 33.17, 24: 56.06, 25: 57.29, 26: 114.67,
}


def label_for(afb1):
    return 'Below_Threshold' if afb1 < AFB1_THRESHOLD else 'At_or_Above_Threshold'


def read_cube(bil_path):
    hdr_path = bil_path.with_suffix('.hdr')
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)
    image = envi.open(str(hdr_path), str(bil_path))
    meta = image.metadata
    header_scale = float(meta.get('reflectance scale factor', REFLECTANCE_SCALE))
    if not np.isclose(header_scale, REFLECTANCE_SCALE):
        raise ValueError(f'{bil_path.name}: unexpected scale factor {header_scale}')
    wavelengths = np.asarray(meta['wavelength'], dtype=np.float64)
    # CRITICAL: open_memmap() exposes ORIGINAL uint16 values, without
    # Spectral Python image.load() automatic reflectance scaling.
    cube = image.open_memmap()
    if cube.ndim != 3 or cube.dtype != np.dtype('uint16'):
        raise ValueError(f'{bil_path.name}: unexpected shape/dtype {cube.shape}, {cube.dtype}')
    if cube.shape[2] != len(wavelengths):
        raise ValueError(f'{bil_path.name}: band/wavelength mismatch')
    return cube, wavelengths


def save_plot(filename, title, ylabel, xs, ys, labels=None, vline=None):
    plt.figure(figsize=(12, 6))
    for idx, y in enumerate(ys):
        label = labels[idx] if labels else None
        plt.plot(xs, y, label=label)
    if vline is not None:
        plt.axvline(vline, linestyle='--', label=f'Strongest separation ({vline:.2f} nm)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.title(title)
    if labels or vline is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(RAW_DIR)
    bil_files = sorted(RAW_DIR.rglob('*.bil'))
    if len(bil_files) != 52:
        raise ValueError(f'Expected 52 BIL files, found {len(bil_files)}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    metadata_rows = []
    cube_means = []
    cube_stds = []
    wavelength_reference = None
    levels_seen = {}

    print('PISTACHIO SPECTRAL SIGNATURE ANALYSIS — CORRECTED SCALING')
    for number, bil_path in enumerate(bil_files, 1):
        match = re.fullmatch(r'L(\d{2})_\d+', bil_path.stem)
        if not match:
            raise ValueError(f'Invalid sample name: {bil_path.name}')
        level = int(match.group(1))
        if level not in AFB1_LEVELS:
            raise ValueError(f'Unknown contamination level: {level}')
        levels_seen[level] = levels_seen.get(level, 0) + 1
        afb1 = AFB1_LEVELS[level]
        cube, wavelengths = read_cube(bil_path)
        if wavelength_reference is None:
            wavelength_reference = wavelengths
            if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
                raise ValueError('Invalid wavelength vector')
        elif not np.allclose(wavelength_reference, wavelengths, atol=1e-5, rtol=0):
            raise ValueError(f'Wavelength vector differs: {bil_path.name}')

        height, width, bands = cube.shape
        count = min(SPECTRA_PER_CUBE, height * width)
        selected = rng.choice(height * width, size=count, replace=False)
        rows, cols = np.unravel_index(selected, (height, width))
        raw = np.asarray(cube[rows, cols, :], dtype=np.float32)
        # Exactly ONE conversion, from original stored uint16 integers.
        reflectance = raw / REFLECTANCE_SCALE
        finite_rows = np.isfinite(reflectance).all(axis=1)
        reflectance = reflectance[finite_rows]
        if len(reflectance) < 2:
            raise ValueError(f'Not enough finite spectra: {bil_path.name}')
        cube_means.append(reflectance.mean(axis=0, dtype=np.float64))
        cube_stds.append(reflectance.std(axis=0, ddof=1, dtype=np.float64))
        metadata_rows.append({
            'Sample_ID': bil_path.stem, 'Level': level,
            'AFB1_ug_per_kg': afb1, 'Binary_Class': label_for(afb1),
            'Spectra_Used': len(reflectance), 'Cube_Lines': height,
            'Cube_Samples': width, 'Cube_Bands': bands,
            'Reflectance_Min': float(reflectance.min()),
            'Reflectance_Max': float(reflectance.max()),
        })
        print(f'[{number:02d}/52] {bil_path.stem}: raw max={raw.max():.0f}, '
              f'reflectance max={reflectance.max():.6f}, sampled={len(reflectance)}')

    if len(levels_seen) != 26 or any(levels_seen.get(i) != 2 for i in AFB1_LEVELS):
        raise ValueError(f'Unexpected level distribution: {levels_seen}')
    cube_means = np.asarray(cube_means, dtype=np.float64)
    cube_stds = np.asarray(cube_stds, dtype=np.float64)
    if cube_means.shape != (52, 462) or not np.isfinite(cube_means).all():
        raise ValueError(f'Invalid cube-level spectral matrix: {cube_means.shape}')
    wavelengths = wavelength_reference
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(OUTPUT_DIR / 'pistachio_cube_spectral_statistics.csv', index=False)
    classes = metadata['Binary_Class'].to_numpy()
    below = cube_means[classes == 'Below_Threshold']
    above = cube_means[classes == 'At_or_Above_Threshold']
    below_mean, above_mean = below.mean(axis=0), above.mean(axis=0)
    below_std, above_std = below.std(axis=0, ddof=1), above.std(axis=0, ddof=1)
    difference = np.abs(above_mean - below_mean)
    pooled_std = np.sqrt(((len(below)-1)*below_std**2 + (len(above)-1)*above_std**2)
                         / (len(below)+len(above)-2))
    separation = np.divide(difference, pooled_std, out=np.full_like(difference, np.nan),
                           where=pooled_std > 1e-12)
    bands = np.arange(1, len(wavelengths)+1)
    pd.DataFrame({
        'Band': bands, 'Wavelength_nm': wavelengths,
        'Below_Threshold_Mean': below_mean, 'Below_Threshold_Std': below_std,
        'At_or_Above_Threshold_Mean': above_mean, 'At_or_Above_Threshold_Std': above_std,
        'Absolute_Mean_Difference': difference,
    }).to_csv(OUTPUT_DIR / 'pistachio_threshold_spectral_signatures.csv', index=False)

    level_rows = []
    for level in sorted(AFB1_LEVELS):
        selected = cube_means[metadata['Level'].to_numpy() == level]
        mean, std = selected.mean(axis=0), selected.std(axis=0, ddof=1)
        for idx in range(len(wavelengths)):
            level_rows.append({
                'Level': level, 'AFB1_ug_per_kg': AFB1_LEVELS[level],
                'Binary_Class': label_for(AFB1_LEVELS[level]),
                'Band': int(bands[idx]), 'Wavelength_nm': wavelengths[idx],
                'Mean_Reflectance': mean[idx], 'Std_Reflectance': std[idx],
            })
    pd.DataFrame(level_rows).to_csv(OUTPUT_DIR / 'pistachio_level_spectral_signatures.csv', index=False)

    # Exploratory correlations: 52 cube means, with two cubes sharing each level.
    rho_values, p_values = [], []
    afb1 = metadata['AFB1_ug_per_kg'].to_numpy()
    for band_idx in range(len(wavelengths)):
        rho, p = spearmanr(afb1, cube_means[:, band_idx])
        rho_values.append(rho)
        p_values.append(p)
    analysis = pd.DataFrame({
        'Band': bands, 'Wavelength_nm': wavelengths,
        'Below_Threshold_Mean': below_mean,
        'At_or_Above_Threshold_Mean': above_mean,
        'Absolute_Threshold_Difference': difference,
        'Pooled_Within_Group_Std': pooled_std,
        'Threshold_Separation_Ratio': separation,
        'Spearman_Rho_AFB1': rho_values,
        'Spearman_P_Value': p_values,
        'Absolute_Spearman_Rho': np.abs(rho_values),
    })
    analysis.to_csv(OUTPUT_DIR / 'pistachio_spectral_band_analysis.csv', index=False)
    top_sep = analysis.sort_values('Threshold_Separation_Ratio', ascending=False).head(30)
    top_corr = analysis.sort_values('Absolute_Spearman_Rho', ascending=False).head(30)
    top_sep.to_csv(OUTPUT_DIR / 'pistachio_top_threshold_discriminative_wavelengths.csv', index=False)
    top_corr.to_csv(OUTPUT_DIR / 'pistachio_top_afb1_correlated_wavelengths.csv', index=False)

    best_idx = int(np.nanargmax(separation))
    save_plot('pistachio_threshold_spectral_signatures.png',
              'Pistachio Spectral Signatures by AFB1 Threshold', 'Mean Reflectance',
              wavelengths, [below_mean, above_mean],
              ['Below 8 µg/kg', 'At/Above 8 µg/kg'], wavelengths[best_idx])
    save_plot('pistachio_threshold_difference.png',
              'Pistachio Spectral Difference Across AFB1 Threshold Groups',
              'Absolute Mean Reflectance Difference', wavelengths, [difference])
    save_plot('pistachio_threshold_separation_ratio.png',
              'Pistachio Wavelength-wise Threshold Separation',
              'Between-Group / Within-Group Variability', wavelengths, [separation])
    save_plot('pistachio_afb1_spectral_correlation.png',
              'Pistachio Spectral Correlation with AFB1 Concentration',
              'Spearman Correlation with AFB1', wavelengths, [rho_values])
    plt.figure(figsize=(13, 7))
    for level in sorted(AFB1_LEVELS):
        subset = cube_means[metadata['Level'].to_numpy() == level]
        plt.plot(wavelengths, subset.mean(axis=0), linewidth=1, alpha=0.45)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Mean Reflectance')
    plt.title('Pistachio Spectral Signatures Across 26 AFB1 Levels')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pistachio_all_level_spectral_signatures.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    print('\nTOP THRESHOLD-DISCRIMINATIVE WAVELENGTHS')
    print(top_sep[['Band', 'Wavelength_nm', 'Threshold_Separation_Ratio',
                   'Absolute_Threshold_Difference']].head(15).to_string(index=False))
    print('\nTOP AFB1-CORRELATED WAVELENGTHS')
    print(top_corr[['Band', 'Wavelength_nm', 'Spearman_Rho_AFB1',
                    'Spearman_P_Value']].head(15).to_string(index=False))
    print(f'\nDONE — 52 cubes, 462 bands, outputs: {OUTPUT_DIR}')
    print('Note: AFB1 p-values are exploratory, unadjusted for 462 tests and paired levels.')


if __name__ == '__main__':
    main()
