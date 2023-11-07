import argparse
import os
import shutil
import traceback

import torch
from lightning import Trainer, seed_everything
from my_utils.config import process_config
from pytorch_models.callbacks import get_callbacks
from pytorch_models.loggers import get_loggers
from pytorch_models.models.survival.clam import CLAM_PL_Surv
from pytorch_models.models.survival.dsmil import DSMIL_PL_Surv
from pytorch_models.models.survival.mamil import MAMIL_PL_Surv
from pytorch_models.models.survival.minet import MINet_PL_Surv
from pytorch_models.models.survival.dtfd import DTFD_PL_Surv
from pytorch_models.models.survival.csmil import CSMIL_PL_Surv
from pytorch_models.models.survival.mmil import MMIL_PL_Surv
from pytorch_models.models.survival.transmil import TransMIL_Features_PL_Surv
from pytorch_models.utils.weight_init import init_weights
from torch.utils.data import DataLoader
from wsi_data.datasets.h5_datasets import FeatureDatasetHDF5


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="Output directory to save experiment.",
        required=True,
    )

    argparser.add_argument(
        "--num-gpus-per-node",
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of GPUS per node.",
        required=False,
    )

    argparser.add_argument(
        "--num-nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Number of training nodes.",
        required=False,
    )

    argparser.add_argument(
        "--name",
        dest="name",
        help="Experiment name.",
        required=True,
    )

    argparser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        help="KFold number.",
        required=True,
    )

    argparser.add_argument(
        "--run",
        dest="run",
        help="Experiment number. Used to increment seed.",
        type=int,
        default=None,
        required=False,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    args = argparser.parse_args()
    return args


def get_data(config):
    train_dataset = FeatureDatasetHDF5(
        data_dir=config.dataset.train_folder,
        data_cols=config.dataset.data_cols,
        base_label=config.dataset.base_label,
    )

    val_dataset = FeatureDatasetHDF5(
        data_dir=config.dataset.val_folder,
        data_cols=config.dataset.data_cols,
        base_label=config.dataset.base_label,
    )

    data = dict()

    data["train"] = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=config.trainer.shuffle,
        pin_memory=True,
        prefetch_factor=config.trainer.prefetch_factor,
        persistent_workers=config.trainer.persistent_workers,
        collate_fn=train_dataset.collate if config.model.classifier == "mil" else None,
    )

    data["val"] = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.trainer.prefetch_factor,
        collate_fn=val_dataset.collate if config.model.classifier == "mil" else None,
    )

    return data


def get_multiresolution_method(config):
    multires_aggregation = config.multires_aggregation
    if multires_aggregation is not None and multires_aggregation.features is None:
        multires_aggregation = None
    else:
        multires_aggregation = dict(multires_aggregation)
    return multires_aggregation


def get_model(config, compile=False, fold=None, run=None):
    checkpoint = config.model.checkpoint
    if checkpoint is not None:
        # load pretrained model
        assert (
            fold is not None and run is not None
        ), "Fold and run number must be provided for checkpoint loading."
        checkpoint = os.path.join(
            checkpoint, f"{fold}_fold/checkpoints/version_{run}/final.ckpt"
        )

    if config.model.classifier == "clam":
        if checkpoint is None:
            model = CLAM_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.clam.size,
                gate=config.model.clam.gated,
                dropout=config.model.clam.dropout,
                k_sample=8,
                instance_eval=config.model.clam.instance_eval,
                instance_loss=config.model.clam.instance_loss,
                instance_loss_weight=0.3,
                subtyping=config.model.clam.subtype,
                multibranch=False,
                multires_aggregation=get_multiresolution_method(config),
                linear_feature=config.model.clam.linear_feature,
                attention_depth=config.model.clam.attention_depth,
                classifier_depth=config.model.clam.classifier_depth,
            )
        else:
            model = CLAM_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.clam.size,
                gate=config.model.clam.gated,
                dropout=config.model.clam.dropout,
                k_sample=8,
                instance_eval=config.model.clam.instance_eval,
                instance_loss=config.model.clam.instance_loss,
                instance_loss_weight=0.3,
                subtyping=config.model.clam.subtype,
                multibranch=False,
                multires_aggregation=get_multiresolution_method(config),
                linear_feature=config.model.clam.linear_feature,
                attention_depth=config.model.clam.attention_depth,
                classifier_depth=config.model.clam.classifier_depth,
            )
    elif config.model.classifier == "transmil":
        if checkpoint is None:
            model = TransMIL_Features_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.transmil.size,
                multires_aggregation=config.model.transmil.multires_aggregation,
            )
        else:
            model = TransMIL_Features_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.transmil.size,
                multires_aggregation=config.model.transmil.multires_aggregation,
            )
    elif config.model.classifier == "dsmil":
        if checkpoint is None:
            model = DSMIL_PL_Surv(
                config=config,
                size=config.model.dsmil.size,
                n_classes=config.dataset.num_classes,
                dropout=config.model.dsmil.dropout,
                nonlinear=config.model.dsmil.nonlinear,
                passing_v=config.model.dsmil.passing_v,
                multires_aggregation=config.model.dsmil.multires_aggregation,
            )
        else:
            model = DSMIL_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                size=config.model.dsmil.size,
                n_classes=config.dataset.num_classes,
                dropout=config.model.dsmil.dropout,
                nonlinear=config.model.dsmil.nonlinear,
                passing_v=config.model.dsmil.passing_v,
                multires_aggregation=config.model.dsmil.multires_aggregation,
            )
    elif config.model.classifier == "mamil":
        if checkpoint is None:
            model = MAMIL_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.mamil.size,
                dropout=config.model.mamil.dropout,
                multires_aggregation=config.model.mamil.multires_aggregation,
            )
        else:
            model = MAMIL_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.mamil.size,
                dropout=config.model.mamil.dropout,
                multires_aggregation=config.model.mamil.multires_aggregation,
            )
    elif "minet" in config.model.classifier:
        if checkpoint is None:
            model = MINet_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.minet.size,
                dropout=config.model.minet.dropout,
                pooling_mode=config.model.minet.pooling_mode,
                multires_aggregation=config.model.minet.multires_aggregation,
            )
        else:
            print("Loading checkpoint")
            model = MINet_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.minet.size,
                dropout=config.model.minet.dropout,
                pooling_mode=config.model.minet.pooling_mode,
                multires_aggregation=config.model.minet.multires_aggregation,
            )
    elif config.model.classifier == "dtfd":
        if checkpoint is None:
            model = DTFD_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.dtfd.size,
                K=config.model.dtfd.K,
                n_bags=config.model.dtfd.n_bags,
                dropout=config.model.dtfd.dropout,
                multires_aggregation=config.model.dtfd.multires_aggregation,
            )
        else:
            print("Loading checkpoint")
            model = DTFD_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.dtfd.size,
                K=config.model.dtfd.K,
                n_bags=config.model.dtfd.n_bags,
                dropout=config.model.dtfd.dropout,
                multires_aggregation=config.model.dtfd.multires_aggregation,
            )
    elif config.model.classifier == "csmil":
        if checkpoint is None:
            model = CSMIL_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.csmil.size,
                cluster_num=config.model.csmil.cluster_num,
                multires_aggregation=config.model.csmil.multires_aggregation,
            )
        else:
            print("Loading checkpoint")
            model = CSMIL_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.csmil.size,
                cluster_num=config.model.csmil.cluster_num,
                multires_aggregation=config.model.csmil.multires_aggregation,
            )
    elif config.model.classifier == "mmil":
        if checkpoint is None:
            model = MMIL_PL_Surv(
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.mmil.size,
                num_msg=config.model.mmil.num_msg,
                num_subbags=config.model.mmil.num_subbags,
                mode=config.model.mmil.mode,
                ape=config.model.mmil.ape,
                num_layers=config.model.mmil.num_layers,
                multires_aggregation=config.model.mmil.multires_aggregation,
            )
        else:
            print("Loading checkpoint")
            model = MMIL_PL_Surv.load_from_checkpoint(
                checkpoint,
                config=config,
                n_classes=config.dataset.num_classes,
                size=config.model.mmil.size,
                num_msg=config.model.mmil.num_msg,
                num_subbags=config.model.mmil.num_subbags,
                mode=config.model.mmil.mode,
                ape=config.model.mmil.ape,
                num_layers=config.model.mmil.num_layers,
                multires_aggregation=config.model.mmil.multires_aggregation,
            )
    else:
        raise ValueError(f"Unknown classifier {config.model.classifier}")

    if compile:
        model = torch.compile(model)
        # model = torch.compile(model, mode="reduce-overhead")

    if config.model.initializer is not None:
        init_weights(model, init_type=config.model.initializer, init_bias=False)

    return model


def get_trainer(config, num_gpus, num_nodes, loggers, callbacks):
    strategy = "auto"  # "ddp" if num_gpus > 1 or num_nodes > 1 else "auto"
    trainer = Trainer(
        accelerator="gpu",
        precision=config.trainer.precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=config.trainer.epochs,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
    )
    return trainer


def rm_ckpt_logs(checkpoint_dir):
    try:
        try:
            version = os.path.basename(checkpoint_dir)
            parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
            shutil.rmtree(os.path.join(parent_dir, "logs", version))
            shutil.rmtree(os.path.join(parent_dir, "checkpoints", version))
        except Exception:
            version = os.path.basename(checkpoint_dir)
            parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
            os.rename(
                os.path.join(parent_dir, "logs", version),
                os.path.join(parent_dir, "logs", version + "_old"),
            )
            os.rename(
                os.path.join(parent_dir, "checkpoints", version),
                os.path.join(parent_dir, "checkpoints", version + "_old"),
            )
    except Exception:
        pass


def main():
    args = get_args()

    config = process_config(
        args.config,
        name=args.name,
        output_dir=args.output_dir,
        fold=args.fold,
        mkdirs=True,
        config_copy=True,
        version=args.run,
    )

    if args.run is not None:
        assert args.run >= 0, "Run number must be >= 0"
        config.trainer.seed = config.trainer.seed + args.run

    # train
    print("Fitting model on datamodule.")
    max_n_errors = 1
    n_errors = 0
    while n_errors < max_n_errors:
        try:
            seed_everything(config.trainer.seed)

            # datasets
            data = get_data(config)

            # model
            model = get_model(
                config, compile=config.model.compile, fold=args.fold, run=args.run
            )
            print(model.model)

            # save initial model
            checkpoint = model.model.state_dict()
            torch.save(
                checkpoint,
                os.path.join(config.callbacks.checkpoint_dir, "initial.ckpt"),
            )

            # loggers
            loggers = get_loggers(config)

            # callbacks
            callbacks = get_callbacks(config)

            # initialize trainer
            trainer = get_trainer(
                config, args.num_gpus, args.num_nodes, loggers, callbacks
            )

            trainer.fit(
                model,
                train_dataloaders=data["train"],
                val_dataloaders=data["val"],
                ckpt_path=os.path.join(config.callbacks.checkpoint_dir, "final.ckpt")
                if config.trainer.resume
                else None,
            )
            break
        except RuntimeError as e:
            traceback.print_exc()
            print("Training failed. Retrying.")
            rm_ckpt_logs(config.callbacks.checkpoint_dir)
            n_errors += 1

    if n_errors == max_n_errors:
        print("Training failed too many times. Exiting.")
        exit(1)

    # validate
    trainer.validate(dataloaders=data["val"], ckpt_path="best", verbose=True)
    trainer.save_checkpoint(os.path.join(config.callbacks.checkpoint_dir, "final.ckpt"))


if __name__ == "__main__":
    main()
