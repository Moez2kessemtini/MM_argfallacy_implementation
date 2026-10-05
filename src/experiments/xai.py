"""Word-level explanations of the AFC text model (six attribution methods) and their evaluation.

Usage:
  python -m src.experiments.xai --seeds 42 2024 666
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from scipy.stats import spearmanr

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.datasets import InputMode, MMUSEDFallacy
from mamkit.models.text import Transformer
from transformers import AutoTokenizer

from src.paths import BASE_DATA_PATH
from src.training.loop import SHARED_TASK_SPLIT, SHARED_TASK_TEST_DIALOGUES
from src.training.runner import load_config, result_dir
from src.utils import macro_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK = 'afc'
CLASSES = ['Appeal to Emotion', 'Appeal to Authority', 'Ad Hominem', 'False Cause', 'Slippery Slope', 'Slogans']
METHODS = ('attention', 'rollout', 'grad_norm', 'grad_x_input', 'ig', 'shapley')
METHOD_LABELS = {'attention': 'Attention (dernière couche)', 'rollout': 'Attention rollout',
                 'grad_norm': 'Norme du gradient', 'grad_x_input': 'Gradient × entrée',
                 'ig': 'Gradients intégrés', 'shapley': 'Shapley (permutations)'}
WORD_RE = re.compile(r'\S+')
# function words left out of the per-class lexicon only (explanations themselves keep every word)
STOPWORDS = set('a an the and or but if of to in on at for with by from as is are was were be been being it its '
                'this that these those there here do does did have has had will would can could should '
                'not no so than then very just also about into over out up down all any some more most '
                "what which who whom when where why how i'm it's that's".split())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--text-run', default='text_only_roberta_ft_imb-weighted_train')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 2024, 666])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--ig-steps', type=int, default=48)
    parser.add_argument('--shapley-perms', type=int, default=32)
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--n-random', type=int, default=10, help='random word sets per snippet and k')
    parser.add_argument('--lexicon-method', choices=METHODS, default='ig')
    parser.add_argument('--min-count', type=int, default=5, help='min occurrences for the lexicon')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--keep-stopwords', action='store_true', help='keep function words in the lexicon')
    return parser.parse_args()


def normalize(word: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", '', word.lower())


def keep_words(words, keep) -> str:
    return ' '.join(w for i, w in enumerate(words) if i in keep) or '.'


class Explainer:
    def __init__(self, model, tokenizer, device):
        self.model, self.tok, self.device = model, tokenizer, device
        self.embeddings = model.model.embeddings.word_embeddings

    # ---------------------------------------------------------------- forward helpers
    @th.no_grad()
    def probs(self, texts, batch_size=128):
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[start:start + batch_size]), truncation=True, max_length=512,
                           padding=True, return_tensors='pt').to(self.device)
            logits = self.model({'inputs': enc['input_ids'], 'input_mask': enc['attention_mask']})
            out.append(th.softmax(logits.float(), -1).cpu())
        return th.cat(out).numpy()

    def _logits_from_embeds(self, embeds, mask):
        hidden = self.model.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        return self.model.head(pooled)

    def encode(self, text):
        enc = self.tok(text, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors='pt')
        words = [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]
        token_word = []
        for s, e in enc['offset_mapping'][0].tolist():
            w_idx = -1
            if e > s:  # special tokens have empty spans
                w_idx = next((w for w, (ws, we, _) in enumerate(words) if s < we and e > ws), -1)
            token_word.append(w_idx)
        return {'ids': enc['input_ids'].to(self.device), 'mask': enc['attention_mask'].to(self.device),
                'words': [w for _, _, w in words], 'token_word': np.array(token_word)}

    @staticmethod
    def to_words(token_scores, enc):
        scores = np.zeros(len(enc['words']))
        for t, w in enumerate(enc['token_word']):
            if w >= 0:
                scores[w] += token_scores[t]
        return scores

    # ---------------------------------------------------------------- methods
    def attention_scores(self, enc):
        with th.no_grad():
            att = self.model.model(input_ids=enc['ids'], attention_mask=enc['mask'], output_attentions=True).attentions
        att = [a[0].float().mean(0) for a in att]  # per layer: [L, L], head-averaged
        last = att[-1][0].cpu().numpy()
        eye = th.eye(att[0].shape[0], device=att[0].device)
        rollout = eye
        for a in att:
            a = 0.5 * a + 0.5 * eye
            rollout = (a / a.sum(-1, keepdim=True)) @ rollout
        return {'attention': self.to_words(last, enc), 'rollout': self.to_words(rollout[0].cpu().numpy(), enc)}

    def gradient_scores(self, enc, target):
        x = self.embeddings(enc['ids']).detach().requires_grad_(True)
        logit = self._logits_from_embeds(x, enc['mask'])[0, target]
        grad, = th.autograd.grad(logit, x)
        return {'grad_norm': self.to_words(grad[0].norm(dim=-1).cpu().numpy(), enc),
                'grad_x_input': self.to_words((grad[0] * x[0]).sum(-1).detach().cpu().numpy(), enc)}

    def ig_scores(self, enc, target, steps):
        x = self.embeddings(enc['ids']).detach()
        special = (enc['ids'] == self.tok.cls_token_id) | (enc['ids'] == self.tok.sep_token_id)
        x0 = self.embeddings(th.where(special, enc['ids'], th.full_like(enc['ids'], self.tok.pad_token_id))).detach()
        with th.no_grad():
            f_x = th.softmax(self._logits_from_embeds(x, enc['mask']).float(), -1)[0, target].item()
            f_0 = th.softmax(self._logits_from_embeds(x0, enc['mask']).float(), -1)[0, target].item()
        alphas = (th.arange(steps, device=self.device, dtype=x.dtype) + 0.5) / steps
        path = (x0 + alphas[:, None, None] * (x - x0)).requires_grad_(True)
        p = th.softmax(self._logits_from_embeds(path, enc['mask'].expand(steps, -1)).float(), -1)[:, target]
        grads, = th.autograd.grad(p.sum(), path)
        token_attr = ((x - x0)[0] * grads.mean(0)).sum(-1).detach().cpu().numpy()
        return self.to_words(token_attr, enc), float(token_attr.sum() - (f_x - f_0))

    def shapley_scores(self, words, target, n_perms, rng):
        n = len(words)
        perms = [rng.permutation(n) for _ in range(n_perms)]
        texts = [keep_words(words, set(perm[:j])) for perm in perms for j in range(n + 1)]
        values = self.probs(texts)[:, target].reshape(n_perms, n + 1)
        phi = np.zeros(n)
        for perm, v in zip(perms, values):
            phi[perm] += np.diff(v)
        return phi / n_perms

    def explain(self, text, methods, ig_steps, n_perms, rng):
        enc = self.encode(text)
        p_full = self.probs([text])[0]
        target = int(p_full.argmax())
        out = {'target': target, 'p': float(p_full[target]), 'words': enc['words'], 'scores': {}}
        if {'attention', 'rollout'} & set(methods):
            out['scores'].update(self.attention_scores(enc))
        if {'grad_norm', 'grad_x_input'} & set(methods):
            out['scores'].update(self.gradient_scores(enc, target))
        if 'ig' in methods:
            out['scores']['ig'], out['ig_gap'] = self.ig_scores(enc, target, ig_steps)
        if 'shapley' in methods:
            out['scores']['shapley'] = self.shapley_scores(enc['words'], target, n_perms, rng)
        out['scores'] = {m: s for m, s in out['scores'].items() if m in methods}
        return out


def faithfulness(explainer, records, methods, ks, n_random, rng):
    rows = []
    for k in ks:
        eligible = [r for r in records if len(r['words']) > k]
        if not eligible:
            continue
        targets = np.array([r['target'] for r in eligible])
        p_full = np.array([r['p'] for r in eligible])
        idx = np.arange(len(eligible))

        def drops(texts, repeat=1):
            p = explainer.probs(texts).reshape(len(eligible), repeat, -1)
            return p_full[:, None] - p[idx, :, targets], p.argmax(-1) != targets[:, None]

        rand_sets = [[set(rng.choice(len(r['words']), k, replace=False)) for _ in range(n_random)] for r in eligible]
        all_idx = [set(range(len(r['words']))) for r in eligible]
        comp_r, flip_r = drops([keep_words(r['words'], all_idx[i] - s) for i, r in enumerate(eligible)
                                for s in rand_sets[i]], n_random)
        suff_r, _ = drops([keep_words(r['words'], s) for i, r in enumerate(eligible) for s in rand_sets[i]], n_random)
        rows.append({'k': k, 'method': 'random', 'n': len(eligible), 'comprehensiveness': comp_r.mean(),
                     'sufficiency': suff_r.mean(), 'flip_rate': flip_r.mean()})
        for m in methods:
            top = [set(np.argsort(-r['scores'][m])[:k]) for r in eligible]
            comp, flip = drops([keep_words(r['words'], all_idx[i] - top[i]) for i, r in enumerate(eligible)])
            suff, _ = drops([keep_words(r['words'], top[i]) for i, r in enumerate(eligible)])
            rows.append({'k': k, 'method': m, 'n': len(eligible), 'comprehensiveness': comp.mean(),
                         'sufficiency': suff.mean(), 'flip_rate': flip.mean(),
                         'comp_gt_random': float((comp[:, 0] > comp_r.mean(1)).mean())})
    return rows


def latex_snippet(words, scores, signed):
    """Colour-coded words: red = towards the predicted class, blue = against (signed methods only)."""
    scale = np.abs(scores).max() or 1.0
    parts = []
    for w, a in zip(words, scores):
        w = re.sub(r'([&%$#_{}])', r'\\\1', w).replace('~', r'\textasciitilde{}').replace('^', r'\^{}')
        level = int(round(60 * abs(a) / scale))
        color = 'blue' if (signed and a < 0) else 'red'
        parts.append(f'\\colorbox{{{color}!{level}}}{{\\strut {w}}}' if level >= 5 else w)
    return ' '.join(parts)


SIGNED = {'grad_x_input', 'ig', 'shapley'}


def main():
    args = parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(0)
    methods = list(args.methods)
    lex_method = args.lexicon_method if args.lexicon_method in methods else methods[0]
    config = load_config(TransformerConfig, ConfigKey(dataset='mmused-fallacy', input_mode=InputMode.TEXT_ONLY,
                                                      task_name=TASK, tags={'anonymous', 'roberta'}))
    tokenizer = AutoTokenizer.from_pretrained(config.model_card)
    afc = MMUSEDFallacy(task_name=TASK, input_mode=InputMode.TEXT_ONLY, base_data_path=BASE_DATA_PATH).data
    test = afc[afc['fallacy'].notna() & afc['dialogue_id'].isin(SHARED_TASK_TEST_DIALOGUES)].reset_index(drop=True)
    texts, labels = test['snippet'].astype(str).tolist(), test['fallacy'].astype(int).to_numpy()

    out = result_dir(SHARED_TASK_SPLIT, TASK, f'xai_{args.text_run}')
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    per_seed, faith_rows, sanity, ig_gaps = {}, [], {}, []
    for seed in args.seeds:
        weights = result_dir(SHARED_TASK_SPLIT, TASK, args.text_run) / f'model_seed{seed}_fold0.pt'
        model = Transformer(model_card=config.model_card, head=config.head, dropout_rate=config.dropout_rate,
                            is_transformer_trainable=False)
        model.load_state_dict(th.load(weights, map_location='cpu'))
        model.to(device).eval()
        explainer = Explainer(model, tokenizer, device)

        records = []
        for i, text in enumerate(texts):
            r = explainer.explain(text, methods, args.ig_steps, args.shapley_perms, rng)
            r.update({'idx': i, 'label': int(labels[i])})
            records.append(r)
            if 'ig_gap' in r:
                ig_gaps.append(abs(r['ig_gap']))
        sanity[seed] = macro_f1(labels, np.array([r['target'] for r in records]))
        logger.info(f'[seed={seed}] test macro F1 {sanity[seed]:.4f} (must equal the text run)')
        for row in faithfulness(explainer, records, methods, args.ks, args.n_random, rng):
            faith_rows.append({'seed': seed, **row})
        per_seed[seed] = records
        del model, explainer
        th.cuda.empty_cache()

    # ---- faithfulness
    faith = pd.DataFrame(faith_rows)
    faith.to_csv(out / 'tables' / 'faithfulness_per_seed.csv', index=False)
    faith_mean = faith.groupby(['k', 'method'])[['comprehensiveness', 'sufficiency', 'flip_rate']].agg(['mean', 'std'])
    faith_mean.to_csv(out / 'tables' / 'faithfulness.csv')

    # ---- agreement between methods (Spearman over words of the same snippet)
    agree = pd.DataFrame(np.nan, index=methods, columns=methods)
    for a, b in combinations(methods, 2):
        rho = [spearmanr(r['scores'][a], r['scores'][b]).correlation
               for rs in per_seed.values() for r in rs if len(r['words']) >= 4]
        agree.loc[a, b] = agree.loc[b, a] = np.nanmean(rho)
    np.fill_diagonal(agree.values, 1.0)
    agree.to_csv(out / 'tables' / 'agreement_spearman.csv')

    # ---- stability across seeds (same method, same snippet)
    stab = []
    for m in methods:
        jac = []
        for a, b in combinations(args.seeds, 2):
            for ra, rb in zip(per_seed[a], per_seed[b]):
                if len(ra['words']) >= 4:
                    ta, tb = set(np.argsort(-ra['scores'][m])[:3]), set(np.argsort(-rb['scores'][m])[:3])
                    jac.append(len(ta & tb) / len(ta | tb))
        stab.append({'method': m, 'top3_jaccard_between_seeds': np.mean(jac) if jac else np.nan})
    same_pred = [np.mean([ra['target'] == rb['target'] for ra, rb in zip(per_seed[a], per_seed[b])])
                 for a, b in combinations(args.seeds, 2)]
    stability = pd.DataFrame(stab)
    stability.to_csv(out / 'tables' / 'stability.csv', index=False)

    # ---- lexicon per predicted class (lexicon method)
    stats = defaultdict(lambda: defaultdict(list))
    for records in per_seed.values():
        for r in records:
            for w, a in zip(r['words'], r['scores'][lex_method]):
                w = normalize(w)
                if w and (args.keep_stopwords or w not in STOPWORDS):
                    stats[r['target']][w].append(a)
    lexicon = pd.DataFrame([{'pred_class': CLASSES[c], 'word': w, 'count': len(v), 'mean_score': float(np.mean(v))}
                            for c in sorted(stats) for w, v in stats[c].items() if len(v) >= args.min_count])
    lexicon = lexicon.sort_values(['pred_class', 'mean_score'], ascending=[True, False])
    lexicon.to_csv(out / 'tables' / f'lexicon_{lex_method}.csv', index=False)
    top_lex = lexicon.groupby('pred_class').head(args.top)

    # ---- confusions and the words behind them
    confusions = defaultdict(list)
    for records in per_seed.values():
        for r in records:
            if r['target'] != r['label']:
                confusions[(r['label'], r['target'])].append(r)
    conf_rows = sorted(((CLASSES[t], CLASSES[p], len(v)) for (t, p), v in confusions.items()), key=lambda x: -x[2])
    pd.DataFrame(conf_rows, columns=['true', 'pred', 'count_over_seeds']).to_csv(out / 'tables' / 'confusions.csv',
                                                                                  index=False)
    error_words = []
    for (t, p), items in sorted(confusions.items(), key=lambda kv: -len(kv[1]))[:3]:
        bag = defaultdict(list)
        for r in items:
            for w, a in zip(r['words'], r['scores'][lex_method]):
                if normalize(w):
                    bag[normalize(w)].append(a)
        ranked = sorted(((w, np.mean(v), len(v)) for w, v in bag.items() if len(v) >= 3), key=lambda x: -x[1])
        error_words.append({'true': CLASSES[t], 'pred': CLASSES[p], 'n': len(items),
                            'top_words': [f'{w} ({m:+.3f}, n={n})' for w, m, n in ranked[:args.top]]})
    with open(out / 'tables' / 'error_words.json', 'w', encoding='utf-8') as f:
        json.dump(error_words, f, indent=2, ensure_ascii=False)

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, title in ((axes[0], 'comprehensiveness', 'Exhaustivité (↑) : suppression des k mots'),
                              (axes[1], 'sufficiency', 'Suffisance (↓) : seuls les k mots gardés')):
        table = faith.groupby(['method', 'k'])[metric].mean().unstack('method')
        for m in ['random'] + methods:
            if m in table:
                ax.plot(table.index, table[m], marker='o', label=METHOD_LABELS.get(m, 'Mots au hasard'),
                        linestyle='--' if m == 'random' else '-', color='grey' if m == 'random' else None)
        ax.set_xlabel('k mots')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('baisse de p(classe prédite)')
    axes[1].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'faithfulness.png', dpi=200)
    plt.close(fig)

    classes_present = [c for c in CLASSES if c in set(top_lex['pred_class'])]
    fig, axes = plt.subplots(1, len(classes_present), figsize=(3.0 * len(classes_present), 3.4), squeeze=False)
    for ax, c in zip(axes[0], classes_present):
        sub = top_lex[top_lex['pred_class'] == c].iloc[::-1]
        ax.barh(np.arange(len(sub)), sub['mean_score'], color='#c0504d')
        ax.set_yticks(np.arange(len(sub)), sub['word'])
        ax.set_title(c, fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
    fig.suptitle(f'Mots les plus attribués ({METHOD_LABELS[lex_method]}), par classe prédite', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'figures' / f'lexicon_{lex_method}.png', dpi=200)
    plt.close(fig)

    # ---- colour-coded examples: same snippet under every method (first seed)
    ref = per_seed[args.seeds[0]]
    picks = [r for r in ref if r['target'] == r['label'] and 6 <= len(r['words']) <= 20][:1]
    if conf_rows:
        t, p = CLASSES.index(conf_rows[0][0]), CLASSES.index(conf_rows[0][1])
        picks += [r for r in ref if r['label'] == t and r['target'] == p and 4 <= len(r['words']) <= 25][:1]
    with open(out / 'examples.tex', 'w', encoding='utf-8') as f:
        for r in picks:
            f.write(f'\\noindent\\textbf{{Vrai : {CLASSES[r["label"]]} ; prédit : {CLASSES[r["target"]]} '
                    f'(p = {r["p"]:.2f})}}\\par\\smallskip\n')
            for m in methods:
                label = METHOD_LABELS[m].replace('×', r'$\times$')
                f.write(f'\\noindent\\makebox[4.4cm][l]{{\\small {label}}} '
                        f'{latex_snippet(r["words"], r["scores"][m], m in SIGNED)}\\par\n')
            f.write('\\medskip\n')

    with open(out / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(f'# XAI -- {args.text_run}, seeds {args.seeds}, methods {methods}\n\n')
        f.write(f'Sanity (test macro F1 of reloaded models): {json.dumps({k: round(v, 4) for k, v in sanity.items()})}\n\n')
        if ig_gaps:
            f.write(f'IG completeness |sum(attr) - (f(x)-f(x0))|: median {np.median(ig_gaps):.4f}, '
                    f'p90 {np.percentile(ig_gaps, 90):.4f} ({args.ig_steps} steps)\n\n')
        f.write('## Faithfulness (mean over seeds; comprehensiveness higher = better, sufficiency lower = better)\n\n'
                + faith_mean.round(4).to_string() + '\n\n')
        f.write('## Agreement between methods (mean Spearman)\n\n' + agree.round(3).to_string() + '\n\n')
        f.write('## Stability across seeds\n\n' + stability.round(3).to_string(index=False) +
                f'\n\nSame predicted class between two seeds: {np.round(same_pred, 3).tolist()}\n\n')
        f.write('## Most frequent confusions (over seeds)\n\n' +
                '\n'.join(f'- {t} -> {p}: {n}' for t, p, n in conf_rows[:6]) + '\n\n')
        f.write(f'## Words behind the top confusions ({lex_method})\n\n' +
                '\n'.join(f'- {e["true"]} -> {e["pred"]} (n={e["n"]}): {", ".join(e["top_words"])}'
                          for e in error_words) + '\n\n')
        f.write(f'## Lexicon ({lex_method}, top words per predicted class)\n\n' + top_lex.round(4).to_string(index=False)
                + '\n')
    logger.info((out / 'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
