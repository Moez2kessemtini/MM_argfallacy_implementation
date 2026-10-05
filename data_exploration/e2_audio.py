"""E2: audio (durations, waveforms, prosodic descriptors per class, normalised per debate)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torchaudio
from scipy import stats

from data_exploration.common import (AFC_LABELS, AFD_LABELS, CLASS_COLORS, LABELS, SPLIT_COLORS, SPLITS, Outputs,
                                     debate_order, load_task, logger)

SAMPLE_RATE = 16000
FRAME_S, HOP_S = 0.040, 0.010          # analysis frames for energy / pitch
F0_MIN, F0_MAX = 65.0, 400.0           # speech F0 search range (Hz)
VOICING_THRESHOLD = 0.5                # normalised autocorrelation peak needed to call a frame voiced
ACTIVE_DB_BELOW_PEAK = 30.0            # frames within 30 dB of the snippet's loud frames count as speech
SHORT_CLIP_S, LONG_CLIP_S = 0.3, 60.0

PROSODY_FEATURES = {
    'duration_s': 'duration (s)',
    'words_per_s': 'speech rate (words/s)',
    'pause_ratio': 'pause ratio',
    'energy_db_mean': 'energy, mean (dB)',
    'energy_db_std': 'energy, std (dB)',
    'f0_median_hz': 'F0 median (Hz)',
    'f0_iqr_st': 'F0 IQR (semitones)',
    'f0_range_st': 'F0 range p5-p95 (semitones)',
    'voiced_ratio': 'voiced ratio',
}
QUALITY_FEATURES = {'snr_proxy_db': 'SNR proxy p95-p10 (dB)', 'clipping_pct': 'clipped samples (%)'}


def clip_id(path) -> str:
    return f'{path.parent.name}/{path.name}'


# --------------------------------------------------------------------------- audio helpers
def load_audio(paths) -> np.ndarray:
    """Mono 16 kHz waveform of one clip or of a list of clips concatenated (as MAMKit does)."""
    paths = [paths] if not isinstance(paths, (list, tuple)) else paths
    chunks = []
    for path in paths:
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        chunks.append(wav.mean(dim=0))
    return th.cat(chunks).numpy() if chunks else np.zeros(0, dtype=np.float32)


def frame_signal(x: np.ndarray) -> np.ndarray:
    """[n_frames, frame] read-only view of x (no copy, so very long clips stay cheap in memory)."""
    frame, hop = int(FRAME_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x.astype(np.float32), frame)[::hop]


def frame_pitch(frames: np.ndarray, block: int = 2048):
    """F0 (Hz) and voicing per frame from the normalised autocorrelation (FFT-based), by blocks of frames."""
    window = np.hanning(frames.shape[1]).astype(np.float32)
    n_fft = 1 << (2 * frames.shape[1] - 1).bit_length()
    lag_min, lag_max = int(SAMPLE_RATE / F0_MAX), int(SAMPLE_RATE / F0_MIN)
    f0, voiced = [], []
    for start in range(0, len(frames), block):
        spectrum = np.fft.rfft(frames[start:start + block] * window, n=n_fft, axis=1)
        acf = np.fft.irfft(np.abs(spectrum) ** 2, n=n_fft, axis=1)[:, :frames.shape[1]]
        acf = acf / np.maximum(acf[:, :1], 1e-12)
        search = acf[:, lag_min:lag_max]
        best = search.argmax(axis=1)
        f0.append(SAMPLE_RATE / (best + lag_min))
        voiced.append(search[np.arange(len(search)), best] >= VOICING_THRESHOLD)
    return np.concatenate(f0), np.concatenate(voiced)


def describe_audio(x: np.ndarray, n_words: int) -> dict:
    duration = len(x) / SAMPLE_RATE
    if duration < FRAME_S:
        return {'duration_s': duration}
    frames = frame_signal(x)
    energy_db = 10 * np.log10(np.concatenate([np.mean(frames[i:i + 4096] ** 2, axis=1)
                                              for i in range(0, len(frames), 4096)]) + 1e-10)
    active = energy_db >= np.percentile(energy_db, 95) - ACTIVE_DB_BELOW_PEAK
    f0, voiced = frame_pitch(frames)
    voiced &= active
    f0_voiced = f0[voiced]
    semitones = 12 * np.log2(f0_voiced / F0_MIN) if len(f0_voiced) else np.array([])
    return {
        'duration_s': duration,
        'words_per_s': n_words / duration,
        'pause_ratio': 1.0 - active.mean(),
        'energy_db_mean': energy_db[active].mean(),
        'energy_db_std': energy_db[active].std(),
        'f0_median_hz': np.median(f0_voiced) if len(f0_voiced) else np.nan,
        'f0_iqr_st': np.subtract(*np.percentile(semitones, [75, 25])) if len(semitones) > 4 else np.nan,
        'f0_range_st': np.subtract(*np.percentile(semitones, [95, 5])) if len(semitones) > 4 else np.nan,
        'voiced_ratio': voiced.mean(),
        'snr_proxy_db': np.percentile(energy_db, 95) - np.percentile(energy_db, 10),
        'clipping_pct': 100 * np.mean(np.abs(x) >= 0.99),
    }


# --------------------------------------------------------------------------- 1. durations
def clip_durations(afc: pd.DataFrame, afd: pd.DataFrame, out: Outputs) -> dict:
    paths = {clip_id(p): p for paths in afc['snippet_paths'] for p in paths}
    paths.update({clip_id(p): p for p in afd['sentence_path']})
    cache = out.tables / 'clip_durations.csv'
    if cache.exists():
        durations = pd.read_csv(cache).set_index('clip')['duration_s'].to_dict()
    else:
        durations = {}
        for i, (cid, path) in enumerate(paths.items()):
            try:
                info = torchaudio.info(str(path))
                durations[cid] = info.num_frames / info.sample_rate
            except (RuntimeError, FileNotFoundError):
                durations[cid] = np.nan
            if (i + 1) % 2000 == 0:
                logger.info(f'durations: {i + 1}/{len(paths)} clips')
        pd.DataFrame({'clip': list(durations), 'duration_s': list(durations.values())}).to_csv(cache, index=False)
    return durations


def duration_analysis(afc, afd, durations, out: Outputs):
    out.section('1. Clip durations')
    afc['duration_s'] = [sum(durations.get(clip_id(p), np.nan) for p in paths) for paths in afc['snippet_paths']]
    afd['duration_s'] = [durations.get(clip_id(p), np.nan) for p in afd['sentence_path']]
    n_missing = int(np.isnan(list(durations.values())).sum())
    out.text(f'{len(durations)} distinct clips, {n_missing} missing or unreadable.\n')

    rows = []
    for task, df in (('afc', afc), ('afd', afd)):
        for (split, label), grp in df.groupby(['split', 'label']):
            d = grp['duration_s'].dropna()
            rows.append({'task': task.upper(), 'split': split, 'class': LABELS[task][label], 'n': len(d),
                         'median_s': d.median(), 'mean_s': d.mean(), 'p95_s': d.quantile(0.95), 'max_s': d.max(),
                         f'pct_below_{SHORT_CLIP_S}s': 100 * (d < SHORT_CLIP_S).mean(),
                         f'pct_above_{LONG_CLIP_S:.0f}s': 100 * (d > LONG_CLIP_S).mean()})
    table = pd.DataFrame(rows)
    out.table(table, 'durations_per_class')
    out.md_table(table, '{:.2f}')

    extremes = pd.concat([
        afc.nlargest(15, 'duration_s').assign(kind='longest AFC snippets'),
        afd.nsmallest(15, 'duration_s').assign(kind='shortest AFD sentences'),
    ])[['kind', 'dialogue_id', 'split', 'label_name', 'duration_s', 'text']]
    extremes['text'] = extremes['text'].str.slice(0, 100)
    out.table(extremes, 'duration_extremes')
    out.text('Longest AFC snippets:\n')
    out.md_table(extremes[extremes['kind'] == 'longest AFC snippets'].head(8)[['dialogue_id', 'label_name',
                                                                                'duration_s', 'text']], '{:.1f}')

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    groups = [afc.loc[afc['label'] == c, 'duration_s'].dropna() for c in range(len(AFC_LABELS))]
    box = axes[0].boxplot(groups, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xticks(range(1, len(AFC_LABELS) + 1))
    axes[0].set_xticklabels(AFC_LABELS, rotation=20, ha='right')
    axes[0].set_ylabel('seconds')
    axes[0].set_title('AFC snippet duration per class (no outliers)')
    bins = np.logspace(np.log10(0.1), np.log10(max(afc['duration_s'].max(), 1.0)), 40)
    for split in SPLITS:
        axes[1].hist(afc.loc[afc['split'] == split, 'duration_s'].dropna(), bins=bins, alpha=0.6, density=True,
                     color=SPLIT_COLORS[split], label=split)
    axes[1].set_xscale('log')
    axes[1].set_title('AFC snippet duration (log scale)')
    axes[1].set_xlabel('seconds')
    axes[1].legend()
    bins = np.logspace(np.log10(0.05), np.log10(max(afd['duration_s'].max(), 1.0)), 40)
    for label, color in zip((0, 1), ('#4C72B0', '#C44E52')):
        axes[2].hist(afd.loc[afd['label'] == label, 'duration_s'].dropna(), bins=bins, alpha=0.55, density=True,
                     color=color, label=AFD_LABELS[label])
    axes[2].set_xscale('log')
    axes[2].set_title('AFD sentence clip duration (log scale)')
    axes[2].set_xlabel('seconds')
    axes[2].legend()
    out.figure(fig, 'durations')


# --------------------------------------------------------------------------- 2. debates
def per_debate_audio(afd, out: Outputs):
    out.section('2. Per-debate audio')
    order = debate_order(afd)
    grp = afd.groupby('dialogue_id')['duration_s']
    table = pd.DataFrame({'dialogue_id': order,
                          'split': afd.groupby('dialogue_id')['split'].first().reindex(order).values,
                          'n_clips': grp.size().reindex(order).values,
                          'total_hours': grp.sum().reindex(order).values / 3600,
                          'median_clip_s': grp.median().reindex(order).values})
    out.table(table, 'per_debate_audio')
    out.text(f'Total sentence audio: {table["total_hours"].sum():.1f} h over {len(order)} debates.\n')
    out.md_table(table, '{:.2f}')


# --------------------------------------------------------------------------- 3. speech rate
def duration_vs_text(afc, afd, out: Outputs):
    out.section('3. Duration vs text length')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (task, df) in zip(axes, (('afc', afc), ('afd', afd))):
        d = df.dropna(subset=['duration_s'])
        d = d[d['duration_s'] > 0]
        words = d['text'].str.split().str.len()
        pearson = stats.pearsonr(words, d['duration_s'])[0]
        spearman = stats.spearmanr(words, d['duration_s'])[0]
        rate = words / d['duration_s']
        rows.append({'task': task.upper(), 'n': len(d), 'pearson_r': pearson, 'spearman_rho': spearman,
                     'words_per_s_median': rate.median(), 'words_per_s_p5': rate.quantile(0.05),
                     'words_per_s_p95': rate.quantile(0.95)})
        sample = d.sample(min(len(d), 4000), random_state=0)
        ax.scatter(sample['text'].str.split().str.len(), sample['duration_s'], s=4, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('words')
        ax.set_ylabel('audio duration (s)')
        ax.set_title(f'{task.upper()}: duration vs words (Spearman rho = {spearman:.2f})')
    table = pd.DataFrame(rows)
    out.table(table, 'duration_vs_text')
    out.md_table(table)
    out.text('A speech rate far outside ~2-4 words/s points to a clip that does not match its text '
             '(see E3 for a direct check).\n')
    out.figure(fig, 'duration_vs_words')


# --------------------------------------------------------------------------- 4. waveforms
def waveforms_and_spectrograms(afc, out: Outputs):
    out.section('4. Waveform and mel spectrogram per class')
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=160, n_mels=64)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    fig, axes = plt.subplots(len(AFC_LABELS), 2, figsize=(15, 2.4 * len(AFC_LABELS)))
    rows = []
    for c, name in enumerate(AFC_LABELS):
        pool = afc[(afc['label'] == c) & (afc['split'] == 'train')].dropna(subset=['duration_s'])
        # the snippet whose duration is closest to the class median, capped at 15 s for readability
        target = min(pool['duration_s'].median(), 15.0)
        row = pool.iloc[(pool['duration_s'] - target).abs().argmin()]
        x = load_audio(row['snippet_paths'])
        t = np.arange(len(x)) / SAMPLE_RATE
        axes[c, 0].plot(t, x, linewidth=0.4, color=CLASS_COLORS[c])
        axes[c, 0].set_ylabel(name, fontsize=9)
        axes[c, 0].set_xlim(0, t[-1] if len(t) else 1)
        spec = to_db(mel(th.from_numpy(x).float())).numpy()
        axes[c, 1].imshow(spec, origin='lower', aspect='auto', cmap='magma',
                          extent=[0, len(x) / SAMPLE_RATE, 0, spec.shape[0]])
        axes[c, 1].grid(False)
        rows.append({'class': name, 'dialogue_id': row['dialogue_id'], 'duration_s': row['duration_s'],
                     'text': row['text']})
    axes[0, 0].set_title('waveform')
    axes[0, 1].set_title('mel spectrogram (64 bands, dB)')
    axes[-1, 0].set_xlabel('seconds')
    axes[-1, 1].set_xlabel('seconds')
    out.figure(fig, 'waveforms_spectrograms')
    table = pd.DataFrame(rows)
    out.table(table, 'waveform_examples')
    for _, row in table.iterrows():
        out.text(f'- **{row["class"]}** ({row["dialogue_id"]}, {row["duration_s"]:.1f} s): {row["text"]}')
    out.text('')


# --------------------------------------------------------------------------- 5. prosody
def prosody(afc, out: Outputs):
    out.section('5. Prosodic and quality descriptors (AFC snippets)')
    cache = out.tables / 'afc_prosody.csv'
    if cache.exists():
        feats = pd.read_csv(cache)
    else:
        rows = []
        for i, row in enumerate(afc.itertuples()):
            try:
                x = load_audio(row.snippet_paths)
                rows.append({'row': row.Index, **describe_audio(x, len(str(row.text).split()))})
            except (RuntimeError, FileNotFoundError):
                rows.append({'row': row.Index})
            if (i + 1) % 200 == 0:
                logger.info(f'prosody: {i + 1}/{len(afc)} snippets')
        feats = pd.DataFrame(rows)
        feats.to_csv(cache, index=False)
    df = afc[['dialogue_id', 'split', 'label', 'label_name']].join(feats.set_index('row'))
    all_features = {**PROSODY_FEATURES, **QUALITY_FEATURES}

    # Class medians (training debates)
    train = df[df['split'] == 'train']
    medians = train.groupby('label_name')[list(all_features)].median().reindex(AFC_LABELS)
    out.table(medians, 'prosody_class_medians', index=True)
    out.text('Class medians, training debates:\n')
    out.md_table(medians.reset_index().rename(columns={'label_name': 'class'}), '{:.2f}')

    # Kruskal-Wallis across classes, raw and z-normalised within debate
    normed = train.copy()
    for col in all_features:
        g = normed.groupby('dialogue_id')[col]
        normed[col] = (normed[col] - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    rows = []
    for col, label in all_features.items():
        for variant, frame in (('raw', train), ('z per debate', normed)):
            samples = [frame.loc[frame['label'] == c, col].dropna() for c in range(len(AFC_LABELS))]
            samples = [s for s in samples if len(s) >= 5]
            n = sum(len(s) for s in samples)
            try:
                h, p = stats.kruskal(*samples)
                epsilon2 = (h - len(samples) + 1) / (n - len(samples))
            except ValueError:  # fewer than 2 classes with data, or a constant feature
                h = p = epsilon2 = np.nan
            rows.append({'feature': label, 'variant': variant, 'H': h, 'p_value': p, 'epsilon_squared': epsilon2})
    tests = pd.DataFrame(rows)
    out.table(tests, 'prosody_kruskal_wallis')
    out.text('Kruskal-Wallis test across the 6 classes (training debates); epsilon^2 = effect size '
             '(~0.01 small, ~0.06 medium, ~0.14 large):\n')
    out.md_table(tests, '{:.4g}')

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, (col, label) in zip(axes.flat, PROSODY_FEATURES.items()):
        groups = [normed.loc[normed['label'] == c, col].dropna() for c in range(len(AFC_LABELS))]
        box = ax.boxplot(groups, patch_artist=True, showfliers=False)
        for patch, color in zip(box['boxes'], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(AFC_LABELS) + 1))
        ax.set_xticklabels([n.replace('Appeal to ', 'App. ') for n in AFC_LABELS], rotation=25, ha='right',
                           fontsize=8)
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(f'{label} (z within debate)', fontsize=10)
    fig.tight_layout(h_pad=2.5)
    out.figure(fig, 'prosody_per_class')

    # Recording quality per debate
    order = debate_order(df)
    quality = df.groupby('dialogue_id')[list(QUALITY_FEATURES) + ['pause_ratio']].median().reindex(order)
    out.table(quality, 'quality_per_debate', index=True)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    colors = [SPLIT_COLORS[s] for s in df.groupby('dialogue_id')['split'].first().reindex(order)]
    ax.bar(np.arange(len(order)), quality['snr_proxy_db'], color=colors)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylabel('median SNR proxy (dB)')
    ax.set_title('Recording quality per debate: frame-energy p95 - p10 (orange = 2024 test debates)')
    out.figure(fig, 'quality_per_debate')
    worst = quality.nsmallest(5, 'snr_proxy_db')
    out.text(f'Lowest SNR-proxy debates: {", ".join(f"{d} ({v:.1f} dB)" for d, v in worst["snr_proxy_db"].items())}.\n')


def main():
    out = Outputs('e2_audio')
    afc, afd = load_task('afc'), load_task('afd')
    durations = clip_durations(afc, afd, out)
    duration_analysis(afc, afd, durations, out)
    per_debate_audio(afd, out)
    duration_vs_text(afc, afd, out)
    waveforms_and_spectrograms(afc, out)
    prosody(afc, out)
    out.write_summary()


if __name__ == '__main__':
    main()
