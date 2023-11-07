import argparse
import os
import h5py
import torch
import pickle
from tqdm import tqdm
from pathlib import Path

from my_utils.config import get_config
from pytorch_models.models.classification.clam import CLAM_PL
from torch.utils.data import DataLoader
from wsi_data.datasets.h5_datasets import FeatureDatasetHDF5


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--ckpt",
        dest="ckpt",
        help="Checkpoint folder.",
        required=True,
    )

    argparser.add_argument(
        "--fold",
        dest="fold",
        help="Fold number.",
        type=int,
        default=0,
        required=False,
    )

    argparser.add_argument(
        "--version",
        dest="version",
        help="Version number.",
        type=int,
        default=0,
        required=False,
    )

    argparser.add_argument(
        "--topk",
        dest="topk",
        help="Number of patches with topk attention scores to display.",
        type=int,
        default=10,
        required=False,
    )

    argparser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="Output directory to save experiment.",
        required=True,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    args = argparser.parse_args()
    return args


def main(config, model_folder: str, fold:int, version:int, topk:int, output_dir:Path):
    feature_dataset = FeatureDatasetHDF5(
        data_dir=config.dataset.test_folder,
        data_cols=config.dataset.data_cols,
        base_label=config.dataset.base_label
    )

    data = DataLoader(
        feature_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
    )

    model_ckpt = os.path.join(model_folder, f'{fold}_fold/checkpoints/version_{version}/final.ckpt')

    if config.model.classifier == "clam":
        model = CLAM_PL.load_from_checkpoint(model_ckpt, strict=False)
        model.instance_eval = False
    else:
        raise ValueError("Classifier not supported.")

    model = model.eval()

    results = dict()

    for d in tqdm(data):
        with torch.no_grad():
            output = model(d)
            att_target = output['attention'][0].cpu().detach().squeeze()
            att_context = output['attention'][1].cpu().detach().squeeze()

            results[d['slide_name'][0]] = {
                'preds': output['preds'].cpu().detach().numpy(),
                'labels': d['labels'].cpu().detach().numpy(),
                'attention': {
                    'target': att_target.numpy(),
                    'context': att_context.numpy(),
                },
                'topk': {
                    'k': topk,
                    'target': torch.topk(att_target, topk)[1].numpy(),
                    'context': torch.topk(att_context, topk)[1].numpy(),
                },
                'topk_smallest': {
                    'k': topk,
                    'target': torch.topk(att_target, topk, largest=False)[1].numpy(),
                    'context': torch.topk(att_context, topk, largest=False)[1].numpy(),
                },
                'topk_30percentile': {
                    'k': topk,
                    'target': torch.topk(att_target[att_target < torch.quantile(att_target, 0.3)], topk)[1].numpy(),
                    'context': torch.topk(att_context[att_context < torch.quantile(att_context, 0.3)], topk)[1].numpy(),
                },
                'topk_50percentile': {
                    'k': topk,
                    'target': torch.topk(att_target[att_target < torch.quantile(att_target, 0.5)], topk)[1].numpy(),
                    'context': torch.topk(att_context[att_context < torch.quantile(att_context, 0.5)], topk)[1].numpy(),
                },
                'topk_70percentile': {
                    'k': topk,
                    'target': torch.topk(att_target[att_target < torch.quantile(att_target, 0.7)], topk)[1].numpy(),
                    'context': torch.topk(att_context[att_context < torch.quantile(att_context, 0.7)], topk)[1].numpy(),
                }
            }
            with h5py.File(os.path.join(config.dataset.test_folder, d['slide_name'][0]), 'r') as f:
                target_key = 'x_' + config.dataset.data_cols.features_target.split('_')[-1]
                context_key = 'x_' + config.dataset.data_cols.features_context.split('_')[-1]

                target_ids = results[d['slide_name'][0]]['topk']['target']
                target_ids.sort()
                context_ids = results[d['slide_name'][0]]['topk']['context']
                context_ids.sort()
                results[d['slide_name'][0]]['topk']['target_patches'] = f[target_key][target_ids]
                results[d['slide_name'][0]]['topk']['context_patches'] = f[context_key][context_ids]

                target_ids = results[d['slide_name'][0]]['topk_smallest']['target']
                target_ids.sort()
                context_ids = results[d['slide_name'][0]]['topk_smallest']['context']
                context_ids.sort()
                results[d['slide_name'][0]]['topk_smallest']['target_patches'] = f[target_key][target_ids]
                results[d['slide_name'][0]]['topk_smallest']['context_patches'] = f[context_key][context_ids]

                target_ids = results[d['slide_name'][0]]['topk_30percentile']['target']
                target_ids.sort()
                context_ids = results[d['slide_name'][0]]['topk_30percentile']['context']
                context_ids.sort()
                results[d['slide_name'][0]]['topk_30percentile']['target_patches'] = f[target_key][target_ids]
                results[d['slide_name'][0]]['topk_30percentile']['context_patches'] = f[context_key][context_ids]

                target_ids = results[d['slide_name'][0]]['topk_50percentile']['target']
                target_ids.sort()
                context_ids = results[d['slide_name'][0]]['topk_50percentile']['context']
                context_ids.sort()
                results[d['slide_name'][0]]['topk_50percentile']['target_patches'] = f[target_key][target_ids]
                results[d['slide_name'][0]]['topk_50percentile']['context_patches'] = f[context_key][context_ids]

                target_ids = results[d['slide_name'][0]]['topk_70percentile']['target']
                target_ids.sort()
                context_ids = results[d['slide_name'][0]]['topk_70percentile']['context']
                context_ids.sort()
                results[d['slide_name'][0]]['topk_70percentile']['target_patches'] = f[target_key][target_ids]
                results[d['slide_name'][0]]['topk_70percentile']['context_patches'] = f[context_key][context_ids]


    with open(output_dir / 'cross_scale_topk_attention_patches.p', 'wb') as pfile:
        pickle.dump(results, pfile, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    args = get_args()
    config, _ = get_config(args.config)
    config.filename = args.config
    o_dir = Path(args.output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    main(config, args.ckpt, args.fold, args.version, args.topk, o_dir)
