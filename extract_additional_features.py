import argparse
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import h5py
import natsort
import numpy as np
import timm
import torch
from my_utils.chat import TelegramNotification
from my_utils.config import get_config
from my_utils.deterministic import seed_everything
from my_utils.multiprocessing import create_gpu_array_task
from pytorch_models.models.ssl_features.ctranspath import ctranspath
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram
from wsi_data.datasets.h5_datasets import Single_H5_Image_Dataset


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        help="Directory containing the pathology slides in h5 format.",
        required=True,
    )

    argparser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        default=0,
        help="Which GPU to use.",
        required=False,
    )

    argparser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of available GPU to use.",
        required=False,
    )

    argparser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Seed for deterministic output.",
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


def create_embeddings_hdf5_dataset(
    h5_path,
    embeddings_sizes: Dict[str, int] = None,
    embeddings: np.ndarray = None,
):
    assert Path(h5_path).exists(), "HDF5 file does not exist."

    if embeddings_sizes is not None and embeddings is not None:
        h5_f = h5py.File(name=h5_path, mode="a")
    else:
        h5_f = h5py.File(name=h5_path, mode="r")

    d_embeddings = dict() if embeddings_sizes is not None else None

    spacings = h5_f.keys()

    p = re.compile("^x_*")
    spacings = [s for s in spacings if p.match(s)]

    size = h5_f[spacings[0]].shape[0]

    embedding_keys = list(embeddings_sizes.keys())

    for key in spacings:
        if embeddings_sizes is not None:
            for key_2 in embedding_keys:
                dataset_key = "embeddings_" + key_2 + "_" + key
                if embeddings is not None:
                    h5_f.create_dataset(
                        dataset_key,
                        data=embeddings[key_2][key],
                    )
                else:
                    if key_2 not in d_embeddings:
                        d_embeddings[key_2] = dict()
                    if dataset_key not in h5_f.keys():
                        d_embeddings[key_2][key] = np.zeros(
                            (size, embeddings_sizes[key_2]), dtype=np.float32
                        )
                    else:
                        raise KeyError("Embeddings already exist.")

    h5_f.close()

    return d_embeddings


def get_models(model_names=None, ckpt_dir=None):
    def load_kimianet(ckpt_path):
        model = timm.create_model("densenet121", pretrained=True, num_classes=0)
        w = torch.load(ckpt_path, map_location="cpu")
        w = {k.replace("module.", ""): v for k, v in w.items()}
        w = {k.replace("model.0.", "features."): v for k, v in w.items()}
        model.load_state_dict(w, strict=False)
        return model

    def load_ctranspath(pth):
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(pth)
        model.load_state_dict(td["model"], strict=True)
        return model

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    models = dict()

    if "resnet50_imagenet" in model_names:
        # ImageNet Truncated ResNet50
        resnet50_imagenet = timm.create_model(
            "resnet50", pretrained=True, num_classes=0
        )
        resnet50_imagenet.layer4 = torch.nn.Identity()
        resnet50_imagenet = resnet50_imagenet.to(device)
        resnet50_imagenet.eval()
        models["resnet50_imagenet"] = resnet50_imagenet

    if "resnest50_imagenet" in model_names:
        # ImageNet Truncated ResNest50
        resnest50_imagenet = timm.create_model(
            "resnest50d", pretrained=True, num_classes=0
        )
        resnest50_imagenet.layer4 = torch.nn.Identity()
        resnest50_imagenet = resnest50_imagenet.to(device)
        resnest50_imagenet.eval()
        models["resnest50_imagenet"] = resnest50_imagenet

    if "densenet121_imagenet" in model_names:
        # ImageNet Truncated DenseNet121
        densenet121_imagenet = timm.create_model(
            "densenet121", pretrained=True, num_classes=0
        )
        densenet121_imagenet = densenet121_imagenet.to(device)
        densenet121_imagenet.eval()
        models["densenet121_imagenet"] = densenet121_imagenet

    if "kimianet" in model_names:
        kimianet = load_kimianet(os.path.join(ckpt_dir, "KimiaNet.pth"))
        kimianet = kimianet.to(device)
        kimianet.eval()
        models["kimianet"] = kimianet

    if "ctranspath" in model_names:
        ctp = load_ctranspath(os.path.join(ckpt_dir, "ctranspath.pth"))
        ctp = ctp.to(device)
        ctp.eval()
        models["ctranspath"] = ctp

    return models


def eval_transforms(pretrained=False, inhouse=False):
    if pretrained and not inhouse:
        # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        mean, std = 255 * np.array([0.485, 0.456, 0.406]), 255 * np.array(
            [0.229, 0.224, 0.225]
        )
    elif pretrained and inhouse:
        mean, std = 255 * np.array([0.827, 0.635, 0.853]), 255 * np.array(
            [0.189, 0.238, 0.121]
        )
    else:
        # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        mean, std = 255 * np.array([0.5, 0.5, 0.5]), 255 * np.array([0.5, 0.5, 0.5])
    trnsfrms_val = transforms.Compose(
        # [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        [transforms.Normalize(mean=mean, std=std)]
    )
    return trnsfrms_val


def compute_embeddings(model, x_cpu, transform=None):
    x = x_cpu
    if len(x.shape) == 4 and x.shape[1] not in [1, 3]:
        x = x.permute(0, 3, 1, 2)
    elif x.shape[1] not in [1, 3]:
        x = x.permute(2, 0, 1)
    if transform is not None:
        x = transform(x.float())
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        x = x.to(device)
    y = model.forward(x).detach().cpu().numpy()
    y = np.squeeze(y)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    return y


def batch_compute_embeddings(x_batch, ssl_models, transform=None):
    if ssl_models is not None:
        embeddings = dict()
        for key, model in ssl_models.items():
            with torch.no_grad():
                embeddings[key] = compute_embeddings(
                    ssl_models[key], x_batch, transform
                )
    else:
        embeddings = None

    return embeddings


def process_slide(
    h5_path,
    options,
    ssl_models,
    transform,
    telegramBot,
    telegram_key,
    telegram_id,
):
    image_name = h5_path.stem

    try:
        d_embeddings = create_embeddings_hdf5_dataset(
            h5_path, embeddings_sizes=options["embeddings_sizes"]
        )
    except AssertionError:
        telegramBot.send(f"File {h5_path} does not exists, skipping..")
        return
    except KeyError:
        telegramBot.send(
            f"Some embeddings already exist in HDF5 file. Please remove the specified model from the list. Skipping file..."
        )
        return

    dataset = Single_H5_Image_Dataset(
        h5_file=h5_path,
        image_regex="^x_",
        channels_last=False,
    )

    loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=False)

    telegramBot.send(f"Extracting tiles for image {image_name}")
    if telegram_key is not None and telegram_id is not None:
        pbar = tqdm_telegram(
            total=len(dataset),
            token=telegram_id,
            chat_id=telegram_key,
            desc=f"Extracting tiles for image {image_name}",
        )
    else:
        pbar = tqdm(total=len(dataset))

    idx = 0
    for batch in loader:

        for key in batch.keys():
            x_batch = batch[key]

            _batch_size = x_batch.shape[0]

            if ssl_models is not None:
                embeddings = batch_compute_embeddings(x_batch, ssl_models, transform)

                for key_2 in ssl_models.keys():
                    d_embeddings[key_2][key][idx : idx + _batch_size, ...] = embeddings[
                        key_2
                    ]

        idx += _batch_size
        pbar.update(_batch_size)

    dataset.close()

    create_embeddings_hdf5_dataset(
        h5_path,
        embeddings_sizes=options["embeddings_sizes"],
        embeddings=d_embeddings,
    )


def process_slides(
    h5_files: Union[List[Path], List[str]],
    options: Dict[str, Any],
    telegramBot: TelegramNotification = None,
    telegram_key: str = None,
    telegram_id: str = None,
):
    ssl_models = get_models(
        model_names=options["embeddings_sizes"].keys(), ckpt_dir=options["ckpt_dir"]
    )
    transform = eval_transforms(pretrained=True, inhouse=False)

    for h5_idx, h5_file in enumerate(h5_files):
        image_name = Path(h5_file).stem

        telegramBot.send(
            f"Processing image {image_name}, #{h5_idx + 1}/{len(h5_files)}"
        )

        process_slide(
            h5_file,
            options,
            ssl_models,
            transform,
            telegramBot,
            telegram_key,
            telegram_id,
        )


def main():
    args = get_args()

    seed = args.seed if args.seed is not None else np.random.randint(2**32 - 1)
    seed_everything(seed)

    config, _ = get_config(args.config)

    telegramBot = TelegramNotification(
        token=config.telegram.token, chat_id=config.telegram.chat_id
    )

    if args.gpu == 0:
        telegramBot.send(
            f"Feature extraction started with {args.num_gpus} processes",
        )

    h5_files = natsort.natsorted(
        glob.glob(os.path.join(args.dataset_dir, "*.h5")), key=str
    )

    h5_files_sublist, process_idx = create_gpu_array_task(
        h5_files, args.gpu, args.num_gpus
    )

    h5_files_sublist = [Path(h5_file) for h5_file in h5_files_sublist]

    options = {
        "batch_size": config.preprocess.batch_size,
        "ckpt_dir": config.preprocess.ssl_ckpt_dir,
        "num_workers": config.preprocess.num_workers,
        "embeddings_sizes": config.preprocess.additional_pretrained_models,
    }

    process_slides(
        h5_files=h5_files_sublist,
        options=options,
        telegramBot=telegramBot,
        telegram_key=config.telegram.token,
        telegram_id=config.telegram.chat_id,
    )

    telegramBot.send(f"Features extracted for all images for proccess {process_idx}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
