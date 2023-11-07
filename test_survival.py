import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

from lightning import Trainer
from my_utils.config import get_config
from natsort import os_sorted
from pytorch_models.loggers import get_loggers
from pytorch_models.models.survival.clam import CLAM_PL_Surv
from pytorch_models.models.survival.dsmil import DSMIL_PL_Surv
from pytorch_models.models.survival.mamil import MAMIL_PL_Surv
from pytorch_models.models.survival.minet import MINet_PL_Surv
from pytorch_models.models.survival.dtfd import DTFD_PL_Surv
from pytorch_models.models.survival.csmil import CSMIL_PL_Surv
from pytorch_models.models.survival.mmil import MMIL_PL_Surv
from pytorch_models.models.survival.transmil import TransMIL_Features_PL_Surv
from pytorch_models.utils.metrics.metrics import get_metrics
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
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    argparser.add_argument(
        "--mode",
        dest="mode",
        help="Validation or testing set.",
        choices=["val", "test", "predict"],
        required=True,
    )

    args = argparser.parse_args()
    return args


def main(config, model_folder, mode, verbose=True):

    if mode not in ["val", "test", "predict"]:
        raise ValueError("Mode must be either val or test.")

    data = None
    if mode == "test":
        feature_dataset = FeatureDatasetHDF5(
            data_dir=config.dataset.test_folder,
            data_cols=config.dataset.data_cols,
            base_label=config.dataset.base_label,
        )

        data = DataLoader(
            feature_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=0,
            collate_fn=feature_dataset.surv_collate,
        )
    elif mode == "predict":
        del config.dataset.data_cols.labels
        feature_dataset = FeatureDatasetHDF5(
            data_dir=config.dataset.predict_folder,
            data_cols=config.dataset.data_cols,
        )

        data = DataLoader(
            feature_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=0,
            collate_fn=feature_dataset.surv_collate,
        )

    folder = os.path.join(model_folder, "*fold*")
    folds = os_sorted(glob.glob(folder))

    versions = len(os.listdir(os.path.join(folds[0], "checkpoints/")))

    o_folder = config.callbacks.tensorboard_log_dir

    overall_df = pd.DataFrame()
    prediction_dfs = []

    for version in range(versions):
        results = []
        vo_folder = os.path.join(
            o_folder,
            f"version_{version}",
        )
        for idx, fold in enumerate(folds):
            config.callbacks.tensorboard_log_dir = os.path.join(
                vo_folder,
                f"fold_{idx}",
            )
            Path(config.callbacks.tensorboard_log_dir).mkdir(
                parents=True, exist_ok=True
            )

            if mode == "val":
                feature_dataset = FeatureDatasetHDF5(
                    data_dir=os.path.join(
                        config.dataset.val_folder, f"{idx}_fold", "val"
                    ),
                    data_cols=config.dataset.data_cols,
                    base_label=config.dataset.base_label,
                )

                data = DataLoader(
                    feature_dataset,
                    batch_size=config.trainer.batch_size,
                    num_workers=1,
                    collate_fn=feature_dataset.surv_collate,
                )

            model_ckpt = os.path.join(fold, f"checkpoints/version_{version}/final.ckpt")
            if config.model.classifier == "clam":
                model = CLAM_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
                model.instance_eval = False
            elif config.model.classifier == "transmil":
                model = TransMIL_Features_PL_Surv.load_from_checkpoint(
                    model_ckpt, strict=False
                )
            elif config.model.classifier == "dsmil":
                model = DSMIL_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            elif config.model.classifier == "mamil":
                model = MAMIL_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            elif "minet" in config.model.classifier:
                model = MINet_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            elif "csmil" in config.model.classifier:
                model = CSMIL_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            elif "dtfd" in config.model.classifier:
                model = DTFD_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            elif "mmil" in config.model.classifier:
                model = MMIL_PL_Surv.load_from_checkpoint(model_ckpt, strict=False)
            else:
                raise ValueError("Classifier not supported.")
            model = model.eval()
            model.test_metrics = get_metrics(
                config,
                n_classes=config.dataset.num_classes
                if config.dataset.num_classes > 2
                else 1,
                dist_sync_on_step=False,
                mode=mode,
                segmentation=False,
                survival=True,
            ).clone(prefix=f"{mode}_")

            # loggers
            loggers = get_loggers(config)

            # initialize trainer
            trainer = Trainer(
                accelerator="gpu",
                # precision=16,
                devices=1,
                strategy="auto",
                logger=loggers,
            )

            if mode == "val":
                print("Validating model.")
                results.append(
                    trainer.validate(
                        model, dataloaders=data, ckpt_path=None, verbose=verbose
                    )
                )
                tmp_predictions = trainer.predict(
                    model, dataloaders=data, ckpt_path=None
                )
            elif mode == "test":
                print("Testing model.")
                results.append(
                    trainer.test(
                        model, dataloaders=data, ckpt_path=None, verbose=verbose
                    )
                )
                tmp_predictions = trainer.predict(
                    model, dataloaders=data, ckpt_path=None
                )
            elif mode == "predict":
                tmp_predictions = trainer.predict(
                    model, dataloaders=data, ckpt_path=None
                )

            tmp_predictions = [
                [
                    item["preds"].detach().cpu().numpy().squeeze(),
                    item["censor"].detach().cpu().numpy().squeeze().astype(np.uint8),
                    item["survtime"].detach().cpu().numpy().squeeze(),
                    item["slide_name"],
                ]
                for item in tmp_predictions
            ]
            tmp_predictions = np.concatenate(tmp_predictions, axis=1).transpose()
            tmp = pd.DataFrame(
                tmp_predictions, columns=["preds", "censor", "survtime", "slide_name"]
            )
            tmp["idx"] = pd.Series(np.arange(len(tmp_predictions)))
            tmp["fold"] = idx
            tmp["version"] = version
            prediction_dfs.append(tmp)

        if mode in ["val", "test"]:
            df = pd.concat([pd.DataFrame(r) for r in results], axis=0)
            overall_df = pd.concat([overall_df, df], axis=0)
            df.to_csv(
                os.path.join(
                    vo_folder, "{}_metrics_version_{}.csv".format(mode, version)
                ),
                index=False,
            )
            df.describe().to_csv(
                os.path.join(
                    vo_folder, "summary_{}_metrics_version_{}.csv".format(mode, version)
                ),
                index=True,
            )

    if mode in ["val", "test"]:
        overall_df.to_csv(
            os.path.join(o_folder, "{}_metrics_overall.csv".format(mode)), index=False
        )
        overall_df.describe().to_csv(
            os.path.join(o_folder, "summary_{}_metrics_overall.csv".format(mode)),
            index=True,
        )

    predictions_df = pd.concat(prediction_dfs, axis=0)
    predictions_df.to_csv(
        os.path.join(o_folder, "{}_predictions.csv".format(mode)), index=False
    )


if __name__ == "__main__":
    args = get_args()
    model_folder = args.ckpt
    config, _ = get_config(args.config)
    mode = args.mode
    config.callbacks.tensorboard_log_dir = os.path.join(model_folder, f"{mode}_metrics")
    config.filename = args.config

    main(config, model_folder, mode, verbose=True)
