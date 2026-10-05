"""E1: labels and text (class distributions, lengths, per-debate statistics, duplicates, context)."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.datasets import InputMode

from data_exploration.common import (AFC_LABELS, AFD_LABELS, CLASS_COLORS, DATASET_PKL, LABELS, SPLIT_COLORS,
                                     SPLITS, Outputs, debate_order, load_task, logger)
from src.training.runner import load_config

TOKENIZER_CARD = 'roberta-base'
MAX_TOKENS = 512
CONTEXT_WINDOWS = (1, 2, 3)
N_EXAMPLES = 3
SEED = 42


def normalize_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip()


def n_words(texts) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts])


def n_tokens(tokenizer, texts) -> np.ndarray:
    encoded = tokenizer(list(map(str, texts)), add_special_tokens=True, truncation=False)['input_ids']
    return np.array([len(ids) for ids in encoded])


# --------------------------------------------------------------------------- 1. schema
def raw_schema(out: Outputs):
    raw = pd.read_pickle(DATASET_PKL)
    rows = []
    for col in raw.columns:
        values = raw[col]
        example = next((v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
        rows.append({'column': col, 'python_type': type(example).__name__, 'non_null': int(values.notna().sum()),
                     'example': str(example)[:120].replace('\n', ' ')})
    schema = pd.DataFrame(rows)
    out.table(schema, 'raw_dataset_schema')

    out.section('1. Raw dataset schema')
    out.text(f'`dataset.pkl`: {len(raw)} rows x {raw.shape[1]} columns, {raw["dialogue_id"].nunique()} debates.\n')
    out.md_table(schema[['column', 'python_type', 'non_null']])
    speaker_cols = [c for c in raw.columns if re.search(r'speaker|spk|person|candidate', c.lower())]
    out.text(f'Speaker-like columns: {speaker_cols if speaker_cols else "none"}.\n')

    # Two label columns are present: check whether they ever differ.
    if {'fallacy', 'fallacies'} <= set(raw.columns):
        labelled = raw[raw['fallacy'].notna() | raw['fallacies'].notna()]
        differ = labelled[labelled['fallacy'].astype(str) != labelled['fallacies'].astype(str)]
        out.text(f'`fallacy` vs `fallacies`: {len(differ)} of {len(labelled)} labelled rows differ.\n')
        if len(differ):
            pairs = differ.groupby(['fallacy', 'fallacies']).size().reset_index(name='n')
            out.table(pairs, 'fallacy_vs_fallacies')
            out.md_table(pairs.head(15))

    # Word-level timing provided with the transcripts (used later for word-aligned audio).
    if 'snippet_tokens' in raw.columns:
        first = next(t for t in raw['snippet_tokens'] if len(t))
        keys = sorted(first[0][0].keys()) if first and isinstance(first[0], list) and first[0] else []
        out.text(f'`snippet_tokens` / `dialogue_tokens`: per-sentence word lists with fields {keys}.\n')


# --------------------------------------------------------------------------- 2. distributions
def class_distribution(data: dict, out: Outputs):
    out.section('2. Class distributions')
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [3, 1]})
    for ax, task in zip(axes, ('afc', 'afd')):
        df, names = data[task], LABELS[task]
        counts = (df.groupby(['split', 'label']).size().unstack(fill_value=0)
                  .reindex(index=list(SPLITS), columns=range(len(names)), fill_value=0))
        shares = counts.div(counts.sum(axis=1), axis=0)
        table = pd.DataFrame({'class': names})
        for split in SPLITS:
            table[f'{split}_n'] = counts.loc[split].values
            table[f'{split}_%'] = 100 * shares.loc[split].values
        table['total_n'] = counts.sum(axis=0).values
        out.table(table, f'class_distribution_{task}')
        out.text(f'**{task.upper()}** ({len(df)} samples: train {counts.loc["train"].sum()}, '
                 f'test {counts.loc["test"].sum()})\n')
        out.md_table(table, '{:.1f}')

        x, width = np.arange(len(names)), 0.38
        for i, split in enumerate(SPLITS):
            bars = ax.bar(x + (i - 0.5) * width, 100 * shares.loc[split], width, label=split,
                          color=SPLIT_COLORS[split])
            for bar, n in zip(bars, counts.loc[split]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(n), ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('% of split')
        ax.set_title(f'{task.upper()} class distribution (bar labels = counts)')
        ax.legend()
    out.figure(fig, 'class_distribution')


# --------------------------------------------------------------------------- 3. class weights
def class_weights(data: dict, out: Outputs):
    out.section('3. Class weights')
    out.text('Inverse-frequency weights w_c = N / (K * n_c), computed on the training debates only, '
             'next to the weights of the baseline configs (CrossEntropyLoss).\n')
    for task in ('afc', 'afd'):
        train = data[task][data[task]['split'] == 'train']
        names = LABELS[task]
        counts = np.bincount(train['label'], minlength=len(names))
        weights = len(train) / (len(names) * np.maximum(counts, 1))
        config = load_config(TransformerConfig, ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY,
                                                          task_name=task, tags={'anonymous', 'roberta'}))
        config_weights = config.loss_function().weight.numpy()
        table = pd.DataFrame({'class': names, 'train_n': counts, 'weight_train_inverse_freq': weights,
                              'weight_baseline_config': config_weights})
        out.table(table, f'class_weights_{task}')
        out.text(f'**{task.upper()}**\n')
        out.md_table(table)


# --------------------------------------------------------------------------- 4. lengths
def text_lengths(data: dict, tokenizer, out: Outputs):
    out.section('4. Text lengths')
    stats_rows = []
    for task in ('afc', 'afd'):
        df = data[task]
        df['n_words'] = n_words(df['text'])
        df['n_tokens'] = n_tokens(tokenizer, df['text'])
        if task == 'afc':
            df['n_sentences'] = df['snippet_sentences'].apply(len)
        for (split, label), grp in df.groupby(['split', 'label']):
            stats_rows.append({'task': task.upper(), 'split': split, 'class': LABELS[task][label], 'n': len(grp),
                               'words_median': grp['n_words'].median(), 'words_mean': grp['n_words'].mean(),
                               'words_p95': grp['n_words'].quantile(0.95), 'words_max': grp['n_words'].max(),
                               'tokens_median': grp['n_tokens'].median(), 'tokens_max': grp['n_tokens'].max(),
                               'pct_over_512_tokens': 100 * (grp['n_tokens'] > MAX_TOKENS).mean()})
    stats = pd.DataFrame(stats_rows)
    out.table(stats, 'text_lengths')
    out.md_table(stats[['task', 'split', 'class', 'n', 'words_median', 'words_p95', 'words_max', 'tokens_max',
                        'pct_over_512_tokens']], '{:.1f}')

    afc = data['afc']
    sentences = afc['n_sentences'].value_counts().sort_index()
    out.text(f'AFC snippets spanning more than one sentence: {100 * (afc["n_sentences"] > 1).mean():.1f}% '
             f'(max {afc["n_sentences"].max()} sentences).\n')
    out.table(sentences.rename_axis('n_sentences').reset_index(name='n_snippets'), 'afc_sentences_per_snippet')

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    groups = [afc.loc[afc['label'] == c, 'n_words'] for c in range(len(AFC_LABELS))]
    box = axes[0].boxplot(groups, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xticks(range(1, len(AFC_LABELS) + 1))
    axes[0].set_xticklabels(AFC_LABELS, rotation=20, ha='right')
    axes[0].set_title('AFC snippet length per class (words, no outliers)')
    afd = data['afd']
    bins = np.arange(0, 101, 2)
    for label, color in zip((0, 1), ('#4C72B0', '#C44E52')):
        axes[1].hist(afd.loc[afd['label'] == label, 'n_words'].clip(upper=100), bins=bins, density=True, alpha=0.55,
                     color=color, label=AFD_LABELS[label])
    axes[1].set_title('AFD sentence length (words, clipped at 100)')
    axes[1].set_xlabel('words')
    axes[1].legend()
    axes[2].bar(np.arange(len(sentences)), sentences.values, color='#55A868')
    axes[2].set_xticks(np.arange(len(sentences)))
    axes[2].set_xticklabels(sentences.index.astype(str))
    axes[2].set_title('AFC: sentences per snippet')
    axes[2].set_xlabel('sentences')
    out.figure(fig, 'text_lengths')


# --------------------------------------------------------------------------- 5. debates
def per_debate(data: dict, out: Outputs):
    out.section('5. Per-debate statistics')
    afc, afd = data['afc'], data['afd']
    order = debate_order(afd)
    afc_counts = afc.groupby(['dialogue_id', 'label']).size().unstack(fill_value=0).reindex(
        index=order, columns=range(len(AFC_LABELS)), fill_value=0)
    table = pd.DataFrame({
        'dialogue_id': order,
        'year': [int(d.split('_')[-1]) for d in order],
        'split': ['test' if d in set(afd.loc[afd['split'] == 'test', 'dialogue_id']) else 'train' for d in order],
        'afd_sentences': afd.groupby('dialogue_id').size().reindex(order).values,
        'afd_fallacy_rate_%': 100 * afd.groupby('dialogue_id')['label'].mean().reindex(order).values,
        'afc_snippets': afc_counts.sum(axis=1).values,
        'afc_classes_present': (afc_counts > 0).sum(axis=1).values,
    })
    for c, name in enumerate(AFC_LABELS):
        table[name] = afc_counts[c].values
    out.table(table, 'per_debate')
    out.md_table(table[['dialogue_id', 'split', 'afd_sentences', 'afd_fallacy_rate_%', 'afc_snippets',
                        'afc_classes_present']], '{:.1f}')
    train = table[table['split'] == 'train']
    out.text(f'Training debates: fallacy rate {train["afd_fallacy_rate_%"].min():.1f}-'
             f'{train["afd_fallacy_rate_%"].max():.1f}% (mean {train["afd_fallacy_rate_%"].mean():.1f}%); '
             f'AFC classes present per debate: {train["afc_classes_present"].min()}-'
             f'{train["afc_classes_present"].max()} (median {train["afc_classes_present"].median():.0f}); '
             f'debates with a single AFC class: {(train["afc_classes_present"] == 1).sum()}.\n')
    corr = np.corrcoef(train['year'], train['afd_fallacy_rate_%'])[0, 1]
    out.text(f'Correlation year vs fallacy rate (training debates): r = {corr:.2f}.\n')

    colors = [SPLIT_COLORS[s] for s in table['split']]
    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].bar(x, table['afd_fallacy_rate_%'], color=colors)
    axes[0].set_ylabel('AFD fallacy rate (%)')
    axes[0].set_title('Fallacy rate per debate (orange = 2024 test debates)')
    axes[1].bar(x, table['afd_sentences'], color=colors)
    axes[1].set_ylabel('AFD sentences')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(table['dialogue_id'], rotation=90)
    out.figure(fig, 'per_debate_rate_and_size')

    fig, ax = plt.subplots(figsize=(8, 13))
    values = afc_counts.values
    im = ax.imshow(values, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(AFC_LABELS)))
    ax.set_xticklabels(AFC_LABELS, rotation=30, ha='right')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{d} (test)' if s == 'test' else d for d, s in zip(order, table['split'])], fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j]:
                ax.text(j, i, values[i, j], ha='center', va='center', fontsize=7,
                        color='white' if values[i, j] > values.max() / 2 else 'black')
    ax.set_title('AFC snippets per debate and class')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    out.figure(fig, 'afc_debate_class_heatmap')


# --------------------------------------------------------------------------- 6. examples
def examples(data: dict, out: Outputs):
    out.section('6. Examples per class')
    rows = []
    for task in ('afc', 'afd'):
        df = data[task]
        for label, name in enumerate(LABELS[task]):
            pool = df[(df['label'] == label) & (df['split'] == 'train')]
            for _, row in pool.sample(min(N_EXAMPLES, len(pool)), random_state=SEED).iterrows():
                rows.append({'task': task.upper(), 'class': name, 'dialogue_id': row['dialogue_id'],
                             'text': row['text']})
    table = pd.DataFrame(rows)
    out.table(table, 'examples')
    for (task, name), grp in table.groupby(['task', 'class'], sort=False):
        out.text(f'**{task} - {name}**')
        for _, row in grp.iterrows():
            out.text(f'- ({row["dialogue_id"]}) {row["text"]}')
        out.text('')


# --------------------------------------------------------------------------- 7. duplicates
def duplicates(data: dict, out: Outputs):
    out.section('7. Duplicated texts across debates')
    summary_rows = []
    for task in ('afc', 'afd'):
        df = data[task].copy()
        df['norm'] = df['text'].map(normalize_text)
        groups = df.groupby('norm').agg(n_occurrences=('dialogue_id', 'size'),
                                        n_debates=('dialogue_id', 'nunique'),
                                        debates=('dialogue_id', lambda s: sorted(set(s))),
                                        splits=('split', lambda s: sorted(set(s))),
                                        labels=('label', lambda s: sorted(set(s))))
        cross = groups[groups['n_debates'] > 1].copy()
        cross['label_conflict'] = cross['labels'].apply(len) > 1
        cross['train_test_overlap'] = cross['splits'].apply(lambda s: s == ['test', 'train'])
        cross = cross.sort_values('n_debates', ascending=False).reset_index()
        out.table(cross, f'duplicates_across_debates_{task}')
        test_texts = set(df.loc[df['split'] == 'test', 'norm'])
        train_texts = set(df.loc[df['split'] == 'train', 'norm'])
        overlap = test_texts & train_texts
        n_test_rows_seen = int(df[(df['split'] == 'test') & df['norm'].isin(overlap)].shape[0])
        summary_rows.append({
            'task': task.upper(),
            'distinct_texts_in_2+_debates': len(cross),
            'rows_concerned': int(cross['n_occurrences'].sum()),
            'with_label_conflict': int(cross['label_conflict'].sum()),
            'distinct_texts_train_and_test': len(overlap),
            'test_rows_seen_in_train': n_test_rows_seen,
            'test_rows_total': int((df['split'] == 'test').sum()),
        })
        if len(cross):
            top = cross.head(10)[['norm', 'n_debates', 'labels', 'splits']].copy()
            top['norm'] = top['norm'].str.slice(0, 80)
            out.text(f'**{task.upper()}: most repeated texts**\n')
            out.md_table(top)
    summary = pd.DataFrame(summary_rows)
    out.table(summary, 'duplicates_summary')
    out.md_table(summary)


# --------------------------------------------------------------------------- 8. context
def context_lengths(tokenizer, out: Outputs):
    out.section('8. Dialogue context length (n previous sentences + target text)')
    rows = []
    for task in ('afc', 'afd'):
        for k in CONTEXT_WINDOWS:
            df = load_task(task, context_window=k)
            target = n_tokens(tokenizer, df['text'])
            context = n_tokens(tokenizer, df['context_text']) - 2  # drop the context's own <s> </s>
            total = target + np.maximum(context, 0) + 1  # one extra separator
            empty = (df['context_text'].fillna('').str.len() == 0).mean()
            rows.append({'task': task.upper(), 'context_window': k, 'context_tokens_median': np.median(context),
                         'context_tokens_p95': np.percentile(context, 95), 'total_tokens_median': np.median(total),
                         'total_tokens_p95': np.percentile(total, 95), 'total_tokens_max': total.max(),
                         'pct_total_over_512': 100 * (total > MAX_TOKENS).mean(), 'pct_empty_context': 100 * empty})
    table = pd.DataFrame(rows)
    out.table(table, 'context_lengths')
    out.md_table(table, '{:.1f}')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for task, marker in (('AFC', 'o'), ('AFD', 's')):
        sub = table[table['task'] == task]
        ax.plot(sub['context_window'], sub['total_tokens_median'], marker=marker, label=f'{task} median')
        ax.plot(sub['context_window'], sub['total_tokens_p95'], marker=marker, linestyle='--', label=f'{task} p95')
    ax.axhline(MAX_TOKENS, color='grey', linestyle=':', label='RoBERTa limit (512)')
    ax.set_xticks(CONTEXT_WINDOWS)
    ax.set_xlabel('previous sentences in context (n)')
    ax.set_ylabel('tokens (context + target)')
    ax.set_title('Input length with dialogue context')
    ax.legend(fontsize=8)
    out.figure(fig, 'context_lengths')


def main():
    out = Outputs('e1_text_labels')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CARD)
    data = {task: load_task(task) for task in ('afc', 'afd')}
    logger.info(f'AFC: {len(data["afc"])} samples, AFD: {len(data["afd"])} samples')

    raw_schema(out)
    class_distribution(data, out)
    class_weights(data, out)
    text_lengths(data, tokenizer, out)
    per_debate(data, out)
    examples(data, out)
    duplicates(data, out)
    context_lengths(tokenizer, out)
    out.write_summary()


if __name__ == '__main__':
    main()
