# Détection et classification multimodales de sophismes argumentatifs

Reproduction et extensions des baselines du shared task **MM-ArgFallacy2025**
([Mancini et al., ArgMining @ ACL 2025](https://aclanthology.org/2025.argmining-1.35/)) sur les
débats présidentiels américains, avec [MAMKit](https://github.com/nlp-unibo/mamkit), ainsi qu'une
**méthode proposée de distillation intermodale texte → audio**.

- **AFD** : détection de sophismes (F1 binaire). **AFC** : classification en 6 catégories (F1 macro).
- Entrées : texte, audio, ou texte + audio. Découpage officiel : entraînement sur MM-USED-Fallacy, test sur les deux débats de 2024.
- Chaque configuration est entraînée avec 3 graines (42, 2024, 666) ; les comparaisons reposent sur un bootstrap apparié sur le test.

**Rapport** : [`docs/rapport.pdf`](docs/rapport.pdf) (analyse critique, protocole, résultats et discussion).

## Résultats

Baseline Transformer (RoBERTa / WavLM, fusion tardive), moyenne ± écart-type sur 3 graines.

| Tâche | Modalité | Article | Nous |
|---|---|---|---|
| AFC | Texte | 0,3925 | 0,3885 ± 0,040 |
| AFC | Audio | 0,0643 | 0,1280 ± 0,043 |
| AFC | Texte + audio | 0,3816 | 0,3905 ± 0,016 |
| AFD | Texte | 0,2770 | 0,2711 ± 0,005 |
| AFD | Audio | 0,0000 | 0,0000 ± 0,000 |
| AFD | Texte + audio | 0,2848 | 0,2728 ± 0,004 |

Extensions (AFC sauf mention contraire) :

| Expérience | Résultat |
|---|---|
| **Distillation texte → audio (méthode proposée)** | 0,201 ± 0,031 (baseline audio 0,128) |
| RoBERTa fine-tuné, poids de classes recalculés sur l'entraînement | AFC 0,381 ± 0,043, AFD 0,301 ± 0,012 |
| Audio seul, transcription Whisper → modèle texte | 0,380 ± 0,045 |
| Audio seul, moyenne de WavLM gelé + lecture linéaire | 0,189 ± 0,010 |
| Meilleure fusion vs concaténation (taux d'apprentissage choisi sur la validation) | 0,360 vs 0,391 |
| Contexte dialogique n−1 (moyenne sur la paire / sur la cible seule) | 0,305 / 0,371 |
| Multi-tâche avec la détection de phrases argumentatives | 0,373 ± 0,014 (contrôle 0,364) |

## Méthode proposée : distillation texte → audio

Un élève audio (WavLM gelé + encodeur Transformer) apprend, sans étiquettes, à reproduire la structure de
l'espace d'un enseignant texte (RoBERTa fine-tuné sur les sophismes) sur toutes les phrases alignées des débats
d'entraînement ; à l'inférence, il n'utilise que l'audio. La perte par file de prototypes s'inspire de COMODO
([Chen et al., 2025](https://arxiv.org/abs/2503.07259)). La méthode est évaluée face à plusieurs contrôles
(features brutes, élève supervisé, enseignant générique, efficacité en étiquettes).

```bash
python -m src.experiments.distill --teacher-temp 0.02 --label-fractions 0.1 0.25 0.5 --seeds 42 2024 666
```

## Installation

```bash
conda create -n mm_argfallacy python=3.10 && conda activate mm_argfallacy
pip install -r requirements.txt
conda install -c conda-forge ffmpeg deno     # traitement audio ; runtime JavaScript requis par yt-dlp
```

Les données sont téléchargées et découpées en extraits par MAMKit (37 débats, environ 30 h d'audio) :

```bash
python -m src.data.prepare
```

Données, caractéristiques et résultats sont écrits dans `data/`, `cache/` et `results/`
(modifiables avec `MAMKIT_DATA_PATH` et `MAMKIT_FEATURE_CACHE`).

## Reproduction

```bash
bash scripts/reproduce_baselines.sh      # Tableau 4 : 2 tâches x 3 modalités x 3 graines
bash data_exploration/run_all.sh         # exploration des données et transcriptions Whisper
bash scripts/run_extensions.sh           # toutes les extensions
```

Exécutions individuelles :

```bash
python -m src.experiments.baseline --task afc --modality text --seeds 42 2024 666
python -m src.experiments.baseline --task afc --modality text --finetune --imbalance weighted_train --save-model --seeds 42 2024 666
python -m src.experiments.baseline --task afc --modality text_audio --fusion crossattn --lr 5e-5 --seeds 42 2024 666
python -m src.evaluation.bootstrap --task afc --a <exécution A> --b <exécution B>
```

Chaque exécution écrit `metrics.npy` et les prédictions de test de chaque graine dans
`results/mmused-fallacy/mm-argfallacy-2025/<tâche>/<exécution>/`. Le nom de l'exécution encode
la configuration, par exemple `text_only_roberta_ft_imb-weighted_train_ctx1`.

## Structure du dépôt

```
src/
  data/          préparation des données, cache des caractéristiques audio
  models/        distillation texte → audio, modèles de fusion, contexte dialogique
  training/      boucle d'entraînement, runners, options d'expérience, stratégies de déséquilibre
  experiments/   points d'entrée : baseline, cascade, distill, multitask, xai
  evaluation/    agrégation du Tableau 4, analyse par classe, bootstrap apparié, analyses a posteriori
data_exploration/  E1 étiquettes et texte, E2 audio et prosodie, E3 audit de l'alignement
scripts/           scripts de reproduction de bout en bout
docs/              rapport
```

## Remarques

Par rapport aux démos de MAMKit, le code : évalue sur le découpage officiel du shared task ;
calcule le F1 AFC en moyenne macro ; encode les étiquettes manquantes de sorte que les phrases
non fallacieuses de 2024 soient étiquetées 0 en AFD ; recherche les configurations par égalité ;
et refixe la graine après le prétraitement, pour que les résultats ne dépendent pas du cache.

## Références

```bibtex
@inproceedings{mancini-etal-2025-overview,
  title     = {Overview of {MM}-{A}rg{F}allacy2025 on Multimodal Argumentative Fallacy Detection and Classification in Political Debates},
  author    = {Mancini, Eleonora and Ruggeri, Federico and Villata, Serena and Torroni, Paolo},
  booktitle = {Proceedings of the 12th Workshop on Argument Mining},
  year      = {2025}
}

@inproceedings{mancini-etal-2024-mamkit,
  title     = {{MAMK}it: A Comprehensive Multimodal Argument Mining Toolkit},
  author    = {Mancini, Eleonora and Ruggeri, Federico and Colamonaco, Stefano and Zecca, Andrea and Marro, Samuele and Torroni, Paolo},
  booktitle = {Proceedings of the 11th Workshop on Argument Mining},
  year      = {2024}
}
```
