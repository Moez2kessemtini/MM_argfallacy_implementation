"""Train and evaluate a baseline or one of its variants (fine-tuning, imbalance, context, fusion).

Usage:
  python -m src.experiments.baseline --task afc --modality text --seeds 42 2024 666
"""
import argparse

from src.models.fusion import FUSION_TYPES
from src.models.text_context import CONTEXT_POOLINGS
from src.training.imbalance import IMBALANCE_STRATEGIES
from src.training.loop import SHARED_TASK_SPLIT
from src.training.options import FINETUNE_LR, ExperimentOptions
from src.training.runner import SPLIT_KEYS, run_audio_only, run_text_audio, run_text_only

MODALITY_FN = {
    'text': run_text_only,
    'audio': run_audio_only,
    'text_audio': run_text_audio,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=['afc', 'afd'], required=True)
    parser.add_argument('--modality', choices=list(MODALITY_FN), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override the config default seed(s), e.g. --seeds 42 2024 666')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override the default max_epochs (50 for text, 20 for audio/text_audio)')
    parser.add_argument('--split', choices=SPLIT_KEYS, default=SHARED_TASK_SPLIT,
                        help='mm-argfallacy-2025 = official shared-task split (Table 4, test on 2024 debates)')
    parser.add_argument('--save-model', action='store_true',
                        help='also save the best weights of each seed as model_seed<S>_fold<F>.pt in the run '
                             'directory (needed to use a fine-tuned text model as COMODO teacher)')

    variants = parser.add_argument_group('experiment variants (defaults = Table 4 baseline)')
    variants.add_argument('--fusion', choices=FUSION_TYPES, default='concat',
                          help='text_audio only. concat = baseline; early | intermediate | late | selfattn | '
                               'crossattn = src/models/fusion.py; textonly = control (text path of late)')
    variants.add_argument('--finetune', action='store_true',
                          help=f'train the text encoder (RoBERTa) instead of freezing it; lr {FINETUNE_LR:g} '
                               f'unless --lr is given (text and text_audio)')
    variants.add_argument('--lr', type=float, default=None, help='explicit learning rate')
    variants.add_argument('--imbalance', choices=IMBALANCE_STRATEGIES, default='weighted',
                          help='weighted = config class weights (baseline); weighted_train = inverse-frequency '
                               'weights of the training split; focal = unweighted focal loss; sampler = '
                               'class-balanced batches + unweighted CE (see src/training/imbalance.py)')
    variants.add_argument('--context', type=int, default=0,
                          help='text only: number of previous dialogue sentences fed with the target as a '
                               'RoBERTa sentence pair (src/models/text_context.py); 0 = text alone (baseline)')
    variants.add_argument('--context-pooling', choices=CONTEXT_POOLINGS, default='all',
                          help='with --context: mean over all tokens of the pair (all, default) or over the '
                               'target tokens only (target)')
    args = parser.parse_args()

    if args.fusion != 'concat' and args.modality != 'text_audio':
        parser.error('--fusion requires --modality text_audio')
    if args.finetune and args.modality == 'audio':
        parser.error('--finetune requires a text encoder (--modality text or text_audio)')
    if args.context and args.modality != 'text':
        parser.error('--context is implemented for --modality text only')
    if args.context_pooling != 'all' and not args.context:
        parser.error('--context-pooling requires --context N > 0')
    return args


def main():
    args = parse_args()
    options = ExperimentOptions(fusion=args.fusion, finetune=args.finetune, lr=args.lr, imbalance=args.imbalance,
                                context=args.context, context_pooling=args.context_pooling)

    kwargs = {'task_name': args.task, 'seeds': args.seeds, 'split_key': args.split, 'options': options,
              'save_model': args.save_model}
    if args.max_epochs is not None:
        kwargs['max_epochs'] = args.max_epochs

    MODALITY_FN[args.modality](**kwargs)


if __name__ == '__main__':
    main()
