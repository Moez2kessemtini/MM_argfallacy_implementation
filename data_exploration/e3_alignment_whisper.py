"""E3: text-audio alignment audit of the AFC snippets with Whisper transcripts."""
import argparse
import difflib
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

from data_exploration.common import (AFC_LABELS, CLASS_COLORS, SPLITS, Outputs, debate_order, load_task, logger)
from data_exploration.e2_audio import SAMPLE_RATE, clip_id, load_audio

CATEGORIES = ('aligned', 'audio_overruns', 'audio_truncated', 'mismatched')
CATEGORY_COLORS = {'aligned': '#55A868', 'audio_overruns': '#DD8452', 'audio_truncated': '#4C72B0',
                   'mismatched': '#C44E52'}
SEMANTIC_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='openai/whisper-medium')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--skip-semantic', action='store_true')
    parser.add_argument('--max-snippets', type=int, default=None, help='debug: transcribe only the first N')
    return parser.parse_args()


# --------------------------------------------------------------------------- text metrics
def words(text: str) -> list:
    text = str(text).lower().replace('-', ' ')
    return re.sub(r"[^\w\s']", ' ', text).split()


def word_error_rate(ref: list, hyp: list) -> float:
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def alignment_scores(ref_text: str, hyp_text: str) -> dict:
    ref, hyp = words(ref_text), words(hyp_text)
    matched = sum(b.size for b in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks())
    return {'ref_words': len(ref), 'hyp_words': len(hyp),
            'recall': matched / len(ref) if ref else np.nan,
            'precision': matched / len(hyp) if hyp else 0.0,
            'wer': word_error_rate(ref, hyp)}


def categorize(recall: float, precision: float, t: float) -> str:
    if recall >= t:
        return 'aligned' if precision >= t else 'audio_overruns'
    return 'audio_truncated' if precision >= t else 'mismatched'


# --------------------------------------------------------------------------- transcription
def snippet_key(paths) -> str:
    return '|'.join(clip_id(p) for p in paths)


def transcribe(afc: pd.DataFrame, path, model_card: str, batch_size: int, max_snippets=None):
    from transformers import pipeline

    done = set()
    if path.exists():
        with path.open(encoding='utf-8') as f:
            done = {json.loads(line)['key'] for line in f if line.strip()}
    todo = afc.drop_duplicates('key')
    todo = todo[~todo['key'].isin(done)]
    if max_snippets is not None:
        todo = todo.head(max(0, max_snippets - len(done)))
    logger.info(f'Whisper: {len(done)} snippets already transcribed, {len(todo)} to go ({model_card})')
    if todo.empty:
        return

    cuda = th.cuda.is_available()
    asr = pipeline('automatic-speech-recognition', model=model_card, chunk_length_s=30,
                   device=0 if cuda else -1, torch_dtype=th.float16 if cuda else th.float32)
    generate_kwargs = {'language': 'english', 'task': 'transcribe'}

    def run(audio, word_timestamps=True):
        inputs = [{'raw': x, 'sampling_rate': SAMPLE_RATE} for x in audio]
        return asr(inputs, batch_size=len(inputs), generate_kwargs=generate_kwargs,
                   return_timestamps='word' if word_timestamps else False)

    def run_robust(audio):
        """(result, status) per item, status in {'words', 'no_words', 'failed'}."""
        try:
            return [(res, 'words') for res in run(audio)]
        except Exception:  # noqa: BLE001 -- any pipeline failure: fall back item by item
            pass
        out = []
        for x in audio:
            try:
                out.append((run([x])[0], 'words'))
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning(f'Whisper: word timestamps failed ({type(e).__name__}: {e}), retrying without them')
            try:
                out.append((run([x], word_timestamps=False)[0], 'no_words'))
            except Exception as e:  # noqa: BLE001
                logger.warning(f'Whisper: transcription failed ({type(e).__name__}: {e}), recording an empty one')
                out.append(({'text': '', 'chunks': []}, 'failed'))
        return out

    rows = list(todo.itertuples())
    n_status = {'no_words': 0, 'failed': 0}
    with path.open('a', encoding='utf-8') as f:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            audio = [load_audio(r.snippet_paths) for r in batch]
            for r, x, (res, status) in zip(batch, audio, run_robust(audio)):
                chunks = [{'word': c['text'].strip(), 'start': c['timestamp'][0], 'end': c['timestamp'][1]}
                          for c in res.get('chunks', [])] if status == 'words' else []
                if status in n_status:
                    n_status[status] += 1
                f.write(json.dumps({'key': r.key, 'dialogue_id': r.dialogue_id, 'audio_s': len(x) / SAMPLE_RATE,
                                    'transcript': res['text'].strip(), 'words': chunks,
                                    'word_timestamps': status == 'words', 'status': status}) + '\n')
            f.flush()
            issues = ', '.join(f'{n} {k}' for k, n in n_status.items() if n)
            logger.info(f'Whisper: {min(start + batch_size, len(rows))}/{len(rows)}' + (f' ({issues})' if issues else ''))


def semantic_similarity(texts_a, texts_b) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL)
    model = AutoModel.from_pretrained(SEMANTIC_MODEL).to(device).eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(list(texts[i:i + 64]), padding=True, truncation=True, max_length=256,
                            return_tensors='pt').to(device)
            with th.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            out.append(th.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu())
        return th.cat(out)

    return (embed(texts_a) * embed(texts_b)).sum(-1).numpy()


# --------------------------------------------------------------------------- analysis
def analyze(afc: pd.DataFrame, transcripts: pd.DataFrame, args, out: Outputs):
    usable = transcripts if 'status' not in transcripts else transcripts[transcripts['status'].fillna('words') != 'failed']
    df = afc.merge(usable[['key', 'transcript', 'audio_s']], on='key', how='inner')
    scores = pd.DataFrame([alignment_scores(r, h) for r, h in zip(df['text'], df['transcript'])])
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df['category'] = [categorize(r, p, args.threshold) for r, p in zip(df['recall'], df['precision'])]
    if not args.skip_semantic:
        df['semantic_sim'] = semantic_similarity(df['text'].astype(str).tolist(),
                                                 df['transcript'].fillna('').astype(str).tolist())
    cols = ['dialogue_id', 'split', 'label_name', 'key', 'audio_s', 'ref_words', 'hyp_words', 'recall', 'precision',
            'wer', 'category', 'text', 'transcript'] + (['semantic_sim'] if 'semantic_sim' in df else [])
    out.table(df[cols], 'alignment_per_snippet')

    out.section('Coverage')
    out.text(f'{len(df)} AFC snippets analysed ({df["key"].nunique()} distinct audio segments), model '
             f'`{args.model}`, threshold {args.threshold}.\n')
    if 'status' in transcripts:
        status = transcripts['status'].fillna('words').value_counts()  # NaN = early records, with word timestamps
        out.text(f'Transcription status: {status.get("words", 0)} with word timestamps, '
                 f'{status.get("no_words", 0)} without (pipeline fallback), {status.get("failed", 0)} failed '
                 f'(empty transcript, excluded below).\n')

    out.section('Alignment categories')
    overall = df['category'].value_counts().reindex(CATEGORIES, fill_value=0)
    table = pd.DataFrame({'category': CATEGORIES, 'n': overall.values, '%': 100 * overall.values / len(df)})
    out.md_table(table, '{:.1f}')
    per_split = (pd.crosstab(df['split'], df['category'], normalize='index') * 100).reindex(
        index=list(SPLITS), columns=list(CATEGORIES), fill_value=0)
    out.text('Per split (% of snippets):\n')
    out.md_table(per_split.reset_index(), '{:.1f}')
    per_class = (pd.crosstab(df['label_name'], df['category'], normalize='index') * 100).reindex(
        index=AFC_LABELS, columns=list(CATEGORIES), fill_value=0)
    out.table(per_class, 'categories_per_class', index=True)
    out.text('Per class (% of snippets):\n')
    out.md_table(per_class.reset_index().rename(columns={'label_name': 'class'}), '{:.1f}')

    out.section('Word-level scores')
    metric_cols = ['recall', 'precision', 'wer'] + (['semantic_sim'] if 'semantic_sim' in df else [])
    quantiles = df[metric_cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T.reset_index().rename(columns={'index': 'metric'})
    quantiles.columns = [str(c) for c in quantiles.columns]
    out.md_table(quantiles)
    by_class = df.groupby('label_name')[metric_cols].median().reindex(AFC_LABELS).reset_index()
    out.text('Median per class:\n')
    out.md_table(by_class.rename(columns={'label_name': 'class'}))

    order = debate_order(df)
    per_debate = (pd.crosstab(df['dialogue_id'], df['category'], normalize='index') * 100).reindex(
        index=order, columns=list(CATEGORIES), fill_value=0)
    per_debate['median_wer'] = df.groupby('dialogue_id')['wer'].median().reindex(order)
    per_debate['n'] = df.groupby('dialogue_id').size().reindex(order)
    out.table(per_debate, 'categories_per_debate', index=True)
    worst = per_debate.nsmallest(5, 'aligned')
    out.text(f'Debates with the lowest share of aligned snippets: '
             f'{", ".join(f"{d} ({v:.0f}%)" for d, v in worst["aligned"].items())}.\n')

    out.section('Examples per category')
    examples = []
    for cat in CATEGORIES:
        pool = df[df['category'] == cat]
        for _, row in pool.sample(min(3, len(pool)), random_state=0).iterrows():
            examples.append({'category': cat, 'dialogue_id': row['dialogue_id'], 'recall': row['recall'],
                             'precision': row['precision'], 'text': row['text'], 'transcript': row['transcript']})
            out.text(f'- **{cat}** ({row["dialogue_id"]}, R={row["recall"]:.2f}, P={row["precision"]:.2f})  \n'
                     f'  TEXT: {row["text"]}  \n  WHISPER: {row["transcript"]}')
    out.table(pd.DataFrame(examples), 'category_examples')

    plot(df, per_class, per_debate, args, out)


def plot(df, per_class, per_debate, args, out: Outputs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sample = df.sample(min(len(df), 3000), random_state=0)
    axes[0].scatter(sample['recall'], sample['precision'], s=6, alpha=0.35,
                    c=[CATEGORY_COLORS[c] for c in sample['category']])
    axes[0].axvline(args.threshold, color='grey', linestyle='--')
    axes[0].axhline(args.threshold, color='grey', linestyle='--')
    axes[0].set_xlabel('recall (annotated words found in transcript)')
    axes[0].set_ylabel('precision (transcript words matching text)')
    axes[0].set_title('Word-level alignment of AFC snippets')
    for cat, color in CATEGORY_COLORS.items():
        axes[0].scatter([], [], color=color, label=cat)
    axes[0].legend(fontsize=8, loc='lower left')
    axes[1].hist(df['wer'].clip(upper=2.0), bins=40, color='#4C72B0')
    axes[1].set_xlabel('WER (clipped at 2)')
    axes[1].set_title('Word error rate of Whisper vs annotated text')
    bottom = np.zeros(len(per_class))
    for cat in CATEGORIES:
        axes[2].bar(range(len(per_class)), per_class[cat], bottom=bottom, color=CATEGORY_COLORS[cat], label=cat)
        bottom += per_class[cat].values
    axes[2].set_xticks(range(len(per_class)))
    axes[2].set_xticklabels(per_class.index, rotation=25, ha='right')
    axes[2].set_ylabel('% of snippets')
    axes[2].set_title('Alignment category per class')
    out.figure(fig, 'alignment_overview')

    fig, ax = plt.subplots(figsize=(16, 4.5))
    bottom = np.zeros(len(per_debate))
    x = np.arange(len(per_debate))
    for cat in CATEGORIES:
        ax.bar(x, per_debate[cat], bottom=bottom, color=CATEGORY_COLORS[cat], label=cat)
        bottom += per_debate[cat].values
    ax.set_xticks(x)
    ax.set_xticklabels(per_debate.index, rotation=90)
    ax.set_ylabel('% of snippets')
    ax.set_title('Alignment category per debate (chronological)')
    ax.legend(fontsize=8, ncol=4)
    out.figure(fig, 'alignment_per_debate')

    if 'semantic_sim' in df:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        groups = [df.loc[df['category'] == c, 'semantic_sim'].dropna() for c in CATEGORIES]
        box = ax.boxplot(groups, patch_artist=True, showfliers=False)
        for patch, cat in zip(box['boxes'], CATEGORIES):
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(CATEGORIES) + 1))
        ax.set_xticklabels(CATEGORIES)
        ax.set_ylabel('cosine similarity')
        ax.set_title('Text / transcript semantic similarity per category')
        out.figure(fig, 'semantic_similarity')


def main():
    args = parse_args()
    out = Outputs('e3_alignment_whisper')
    afc = load_task('afc')
    afc['key'] = afc['snippet_paths'].map(snippet_key)
    transcripts_path = out.tables / 'whisper_afc_transcripts.jsonl'
    transcribe(afc, transcripts_path, args.model, args.batch_size, args.max_snippets)
    transcripts = pd.read_json(transcripts_path, lines=True)
    analyze(afc, transcripts, args, out)
    out.write_summary()


if __name__ == '__main__':
    main()
