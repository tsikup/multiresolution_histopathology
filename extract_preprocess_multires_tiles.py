import argparse
import os
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import natsort
import numpy as np
import pandas as pd
import torch
from he_preprocessing.normalization.stain_norm import StainNormalizer
from my_utils.chat import TelegramNotification, send_noti_to_telegram
from my_utils.config import get_config
from my_utils.deterministic import seed_everything
from my_utils.multiprocessing import create_gpu_array_task
from PIL import Image
from pytorch_models.models.ssl_features.resnets import ResNet50_SimCLR
from pytorch_models.models.ssl_features.vit import ViT
from sourcelib.associations import associate_files
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram
from wsi_data.datasets.wsi_datasets import Single_WSI_Dataset
from wsi_data.normalization import get_channels_sums_from_ndarray
from wsi_data.wholeslidedata.utils import get_files


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--slides-dir",
        dest="slides_dir",
        help="Directory containing the pathology slides.",
        required=True,
    )

    argparser.add_argument(
        "--annotations-dir",
        dest="annotations_dir",
        help="Directory containing the annotation files.",
        required=True,
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to save hdf5 files for each slide.",
        required=True,
    )

    argparser.add_argument(
        "--slide-extension",
        dest="slide_extension",
        default=".ndpi",
        help="Extension of digital pathology slides.",
        required=False,
    )

    argparser.add_argument(
        "--ann-extension",
        dest="ann_extension",
        default=".geojson",
        help="Extension of digital pathology annotations.",
        required=False,
    )

    argparser.add_argument(
        "--labels-csv",
        dest="labels_csv",
        default=None,
        help="CSV file containing the labels.",
        required=False,
    )

    argparser.add_argument(
        "--image-col",
        dest="image_col",
        default=None,
        help="Image name column in labels csv.",
        required=False,
    )

    argparser.add_argument(
        "--label-col",
        dest="label_col",
        default=None,
        help="label name column in labels csv.",
        required=False,
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


INITIAL_SIZE = 10000
ANN_LABEL = {
    "tissue": 1,
    "tumor": 2,
    "in situ": 3,
}


def create_hdf5(
    h5_path,
    batch_shape,
    label,
    spacing,
    segmentation=False,
    embeddings: Union[Dict[str, int], None] = None,
    coordinates: bool = False,
    save_tiles=False,
    label_col="label",
):
    assert not Path(h5_path).exists()

    h5_f = h5py.File(name=h5_path, mode="w")

    images_shape = batch_shape[1:]

    d_x = dict()
    d_y = dict() if segmentation else None
    d_label = None
    d_embeddings = dict() if embeddings is not None else None
    d_coords: Union[Dict, None] = dict() if coordinates else None

    if label is not None:
        d_label = h5_f.create_dataset(
            label_col,
            shape=(1,),
            dtype=np.uint8,
            data=label,
            chunks=None,
            compression=None,
        )

    for key, value in spacing.items():
        h5_f.attrs["spacing_" + key] = value

        if save_tiles:
            d_x[key] = h5_f.create_dataset(
                "x_" + key,
                (INITIAL_SIZE, *images_shape),
                dtype=np.uint8,
                chunks=(1, *images_shape),
                compression=None,
                maxshape=(None, *images_shape),
            )

            if segmentation:
                d_y[key] = h5_f.create_dataset(
                    "y_" + key,
                    (INITIAL_SIZE, *images_shape[:-1]),
                    dtype=np.uint8,
                    chunks=(1, *images_shape[:-1]),
                    compression=None,
                    maxshape=(None, *images_shape[:-1]),
                )

        if embeddings is not None:
            for key_2 in embeddings.keys():
                if key_2 not in d_embeddings:
                    d_embeddings[key_2] = dict()
                d_embeddings[key_2][key] = h5_f.create_dataset(
                    "embeddings_" + key_2 + "_" + key,
                    (INITIAL_SIZE, embeddings[key_2]),
                    dtype=np.float32,
                    chunks=(1, embeddings[key_2]),
                    maxshape=(None, embeddings[key_2]),
                )

    if coordinates is not None:
        d_coords["x"] = h5_f.create_dataset(
            "coords_x",
            (INITIAL_SIZE, 1),
            dtype=np.uint64,
            chunks=(1, 1),
            compression=None,
            maxshape=(None, 1),
        )
        d_coords["y"] = h5_f.create_dataset(
            "coords_y",
            (INITIAL_SIZE, 1),
            dtype=np.uint64,
            chunks=(1, 1),
            compression=None,
            maxshape=(None, 1),
        )

    return (h5_f, d_x, d_y, d_label, d_embeddings, d_coords)


def get_ssl_models(ckpt_dir):
    vit = ViT(arch="small", ckpt=os.path.join(ckpt_dir, "vits_tcga_brca_dino.pt"))
    resnet50 = ResNet50_SimCLR(
        ckpt=os.path.join(ckpt_dir, "resnet50_tcga_brca_simclr.pt")
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        vit = vit.to(device)
        resnet50 = resnet50.to(device)

    vit.eval()
    resnet50.eval()

    return {
        "vit": vit,
        "resnet50": resnet50,
    }


def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = 255 * np.array([0.485, 0.456, 0.406]), 255 * np.array(
            [0.229, 0.224, 0.225]
        )
    else:
        mean, std = 255 * np.array([0.5, 0.5, 0.5]), 255 * np.array([0.5, 0.5, 0.5])
    trnsfrms_val = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
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


def calculate_intensities(x: np.ndarray):
    _sums, _squared_sums = get_channels_sums_from_ndarray(
        x, channels_last=x.shape[-1] in [1, 3], max_value=255.0
    )
    return _sums, _squared_sums


def process_slide(
    image_file,
    annotations,
    wsa,
    h5_path,
    label,
    options,
    ssl_models,
    transform,
    telegramBot,
    telegram_key,
    telegram_id,
):
    image_name = image_file.path.stem

    dataset = Single_WSI_Dataset(
        image_file,
        annotations,
        tile_size=options["tile_size"],
        spacing=options["spacing"]
        if options["multiresolution"]
        else options["spacing"]["target"],
        transform=None,
        filters2apply=dict(options["filters2apply"]),
        blurriness_threshold=options["blurriness_threshold"]
        if options["multiresolution"]
        else options["blurriness_threshold"]["target"],
        tissue_percentage=options["tissue_percentage"]
        if options["multiresolution"]
        else options["tissue_percentage"]["target"],
        segmentation=options["segmentation"],
        wsa=wsa if options["segmentation"] else None,
    )

    if len(dataset) <= 0:
        return

    batch_shape = [options["batch_size"], options["tile_size"], options["tile_size"], 3]
    batch_size = batch_shape[0]

    loader = DataLoader(
        dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=options["num_workers"],
        pin_memory=True if options["num_workers"] > 0 else False,
        prefetch_factor=2,
    )

    try:
        save_tiles = options["save_tiles"]
        (h5_f, d_x, d_y, _, d_embeddings, d_coords) = create_hdf5(
            h5_path,
            batch_shape,
            label=label,
            spacing=options["spacing"],
            segmentation=options["segmentation"],
            embeddings=options["embeddings_sizes"]
            if options["ckpt_dir"] is not None
            else None,
            save_tiles=save_tiles,
            label_col=options["label_col"],
            coordinates=True,
        )
    except AssertionError:
        telegramBot.send(f"File {h5_path} already exists, skipping")
        return

    sums = dict()
    squared_sums = dict()
    for key in d_x.keys():
        sums[key] = np.array([0, 0, 0], dtype=np.float32)
        squared_sums[key] = np.array([0, 0, 0], dtype=np.float32)

    telegramBot.send(f"Extracting tiles for image {image_name}.")
    if telegram_key is not None and telegram_id is not None:
        pbar = tqdm_telegram(
            total=len(dataset),
            token=telegram_id,
            chat_id=telegram_key,
            desc=f"Extracting tiles for image {image_name}",
        )
    else:
        pbar = tqdm(total=len(dataset))

    effective_batch_size = dict()
    effective_emb_batch_size = dict()
    idx: Union[Dict, None] = dict.fromkeys(d_x.keys(), 0)
    idx_coords = 0
    embeddings_idx = (
        dict.fromkeys(d_embeddings[list(d_embeddings.keys())[0]].keys(), 0)
        if d_embeddings is not None
        else None
    )

    for batch in loader:
        if not batch:
            pbar.update(options["batch_size"])
            continue

        for key in batch.keys():
            x_batch = batch[key]["img_array"]
            coords_x, coords_y = batch[key]["x"], batch[key]["y"]
            _sums, _squared_sums = calculate_intensities(x_batch.numpy())
            if options["segmentation"]:
                y_batch = batch[key]["mask_array"]

            sums[key] += _sums
            squared_sums[key] += _squared_sums

            effective_batch_size[key] = coords_x.shape[0]

            if key == "target":
                d_coords["x"][
                    idx_coords : idx_coords + effective_batch_size[key], ...
                ] = coords_x.reshape(-1, 1)
                d_coords["y"][
                    idx_coords : idx_coords + effective_batch_size[key], ...
                ] = coords_y.reshape(-1, 1)
                idx_coords += effective_batch_size[key]

            if save_tiles:
                d_x[key][idx[key] : idx[key] + effective_batch_size[key], ...] = x_batch
                if options["segmentation"]:
                    d_y[key][
                        idx[key] : idx[key] + effective_batch_size[key], ...
                    ] = y_batch

                idx[key] += effective_batch_size[key]

                current_d_len = d_x[key].shape[0]
                if idx[key] >= current_d_len - batch_size:
                    d_x[key].resize(current_d_len + INITIAL_SIZE, axis=0)
                    if options["segmentation"]:
                        d_y[key].resize(current_d_len + INITIAL_SIZE, axis=0)
                    d_coords["x"].resize(current_d_len + INITIAL_SIZE, axis=0)
                    d_coords["y"].resize(current_d_len + INITIAL_SIZE, axis=0)

            if ssl_models is not None:
                embeddings = batch_compute_embeddings(x_batch, ssl_models, transform)
                _key_2 = list(embeddings.keys())[0]
                effective_emb_batch_size[key] = embeddings[_key_2].shape[0]

                for key_2 in ssl_models.keys():
                    d_embeddings[key_2][key][
                        embeddings_idx[key] : embeddings_idx[key]
                        + effective_emb_batch_size[key],
                        ...,
                    ] = embeddings[key_2]

                embeddings_idx[key] += effective_emb_batch_size[key]

                for key_2 in ssl_models.keys():
                    emb_current_d_len = d_embeddings[key_2][key].shape[0]
                    if embeddings_idx[key] >= emb_current_d_len - batch_size:
                        d_embeddings[key_2][key].resize(
                            emb_current_d_len + INITIAL_SIZE, axis=0
                        )

        pbar.update(options["batch_size"])

    pbar.close()

    # _idx = idx[list(idx.keys())[0]]
    _idx = idx_coords

    if _idx == 0 and not np.any(d_x["target"][0, ...]):
        h5_f.close()
        os.remove(h5_path)
        telegramBot.send(f"No multires tiles extracted for {image_name}.")
        return

    # Resize datasets to proper length.
    if d_x is not None:
        for key in d_x.keys():
            d_x[key].resize(_idx, axis=0)

    if d_y is not None:
        for key in d_y.keys():
            d_y[key].resize(_idx, axis=0)

    if d_embeddings is not None:
        for key_2 in d_embeddings.keys():
            for key in d_embeddings[key_2].keys():
                d_embeddings[key_2][key].resize(_idx, axis=0)

    if d_coords is not None:
        for key in d_coords.keys():
            d_coords[key].resize(_idx, axis=0)

    # Save sums and squared sums for standardization calculation later on.
    for key in sums.keys():
        # d_sums[key], d_squared_sums[key] = sums[key], squared_sums[key]
        h5_f.create_dataset(
            "sums_" + key,
            (3,),
            data=sums[key],
            dtype=np.float32,
        )

        h5_f.create_dataset(
            "squared_sums_" + key,
            (3,),
            data=squared_sums[key],
            dtype=np.float32,
        )

    h5_f.create_dataset(
        "num_batches",
        (1,),
        data=np.array([_idx], dtype=np.uint32),
        dtype=np.uint32,
    )

    h5_f.close()


def process_slides(
    output_dir: Path,
    image_files,
    annotation_files,
    raw_annotation_files,
    slide_extension,
    options: Dict[str, Any],
    labels_csv: Union[str, Path, pd.DataFrame],
    image_col: str,
    label_col: str = None,
    telegram_key: str = None,
    telegram_id: str = None,
):
    telegramBot = TelegramNotification(token=telegram_key, chat_id=telegram_id)

    if label_col is not None:
        assert (
            labels_csv is not None
        ), "Labels csv must be provided if label_col is not None"
        labels_df = (
            labels_csv
            if isinstance(labels_csv, pd.DataFrame)
            else pd.read_csv(labels_csv)
        )
        labels_df = labels_df[~labels_df[image_col].isna()]
        labels_df[image_col] = labels_df[image_col].str.rstrip()
        labels_df[image_col] = labels_df[image_col].apply(
            lambda f: Path(f).stem.split("_")[0].upper()
            if f.endswith(slide_extension)
            else f.split("_")[0].upper()
        )

    if options["ckpt_dir"] is not None:
        ssl_models = get_ssl_models(options["ckpt_dir"])
        transform = eval_transforms(pretrained=False)
    else:
        ssl_models = None
        transform = None

    for image_idx, image_file in enumerate(image_files):
        image_name = image_file.path.stem
        patient_id = image_name.split("_")[0].upper()

        telegramBot.send(
            f"Processing image {image_name}, #{image_idx + 1}/{len(image_files)}"
        )

        label = (
            labels_df.loc[labels_df[image_col] == patient_id, label_col].values[0]
            if label_col is not None
            else None
        )

        h5_path = os.path.join(output_dir, image_name + ".h5")

        if options["filters2apply"] is not None:
            if options["filters2apply"]["stain_norm_target"] is not None:
                target_image = np.array(
                    Image.open(options["filters2apply"]["stain_norm_target"])
                )
            else:
                target_image = None
            if options["filters2apply"]["stain_norm"]:
                options["filters2apply"]["stain_normalizer"] = StainNormalizer(
                    luminosity=options["filters2apply"]["stain_norm_luminosity"],
                    method=options["filters2apply"]["stain_norm_method"],
                    target=target_image,
                    dataset_level_stain_csv=options["filters2apply"][
                        "stain_norm_reference"
                    ]["dataset_level"],
                    slide_level_stain_csv=options["filters2apply"][
                        "stain_norm_reference"
                    ]["slide_level"],
                )
            else:
                options["filters2apply"]["stain_normalizer"] = None

        associations = associate_files([image_file], annotation_files, exact_match=True)
        annotations = associations[image_name]["wsa"]
        annotations = [
            ann
            for i in range(len(annotations))
            for ann in annotations[i].open().annotations
        ]
        if raw_annotation_files is not None:
            associations = associate_files(
                [image_file], raw_annotation_files, exact_match=True
            )
            wsa = associations[image_name]["wsa"][0]
        else:
            wsa = None

        process_slide(
            image_file,
            annotations,
            wsa,
            h5_path,
            label,
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

    if args.gpu == 0:
        send_noti_to_telegram(
            f"Multires tiles extraction started with {args.num_gpus} processes",
            TELEGRAM_TOKEN=config.telegram.token,
            TELEGRAM_CHAT_ID=config.telegram.chat_id,
        )

    slide_extension = (
        "." + config.preprocess.slide_extension
        if not config.preprocess.slide_extension.startswith(".")
        else config.preprocess.slide_extension
    )

    ann_extension = (
        "." + config.preprocess.ann_extension
        if not config.preprocess.ann_extension.startswith(".")
        else config.preprocess.ann_extension
    )

    multiresolution = (
        len(config.preprocess.spacing) > 1
        and config.preprocess.spacing.context is not None
    )

    file_type = "mrwsi" if multiresolution else "wsi"

    spacing = (
        dict(config.preprocess.spacing)
        if multiresolution
        else dict(target=config.preprocess.spacing["target"])
    )

    blurriness_threshold = (
        dict(config.preprocess.filters2apply.blurriness_threshold)
        if multiresolution
        else dict(target=config.preprocess.filters2apply.blurriness_threshold.target)
    )

    tissue_percentage = (
        dict(config.preprocess.filters2apply.keep_tile_percentage)
        if multiresolution
        else dict(target=config.preprocess.filters2apply.keep_tile_percentage.target)
    )

    labels = {
        config.preprocess.annotation_type: ANN_LABEL[config.preprocess.annotation_type]
    }
    segmentation = config.preprocess.save_masks
    segmentation_labels = {}
    for label in config.preprocess.segmentation_labels:
        if label == "insitu":
            label = "in situ"
        segmentation_labels[label] = ANN_LABEL[label]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_files(
        slides_dir=args.slides_dir,
        annotations_dir=args.annotations_dir,
        file_type=file_type,
        tile_size=config.preprocess.filters2apply.tileSize,
        labels=labels,
        stride_overlap_percentage=config.preprocess.stride_overlap_percentage,
        intersection_percentage=config.preprocess.intersection_percentage,
        ratio=config.preprocess.ratio,
        slide_extension=slide_extension,
        ann_extension=ann_extension,
        tiled=True,
        return_raw_annotation=segmentation,
        segmentation_labels=segmentation_labels,
    )

    if len(files) == 2:
        image_files, annotation_files = files
        raw_annotation_files = None
    elif len(files) == 3:
        image_files, annotation_files, raw_annotation_files = files
    else:
        raise ValueError(f"Unexpected number of files returned: {len(files)}")

    image_files = natsort.natsorted(image_files, key=str)

    image_files_sublist, process_idx = create_gpu_array_task(
        image_files, args.gpu, args.num_gpus
    )

    options = {
        "save_tiles": config.preprocess.save_tiles,
        "file_type": file_type,
        "tile_size": config.preprocess.filters2apply.tileSize,
        "batch_size": config.preprocess.batch_size,
        "tissue_percentage": tissue_percentage,
        "stride_overlap_percentage": config.preprocess.stride_overlap_percentage,
        "intersection_percentage": config.preprocess.intersection_percentage,
        "blurriness_threshold": blurriness_threshold,
        "labels": labels,
        "spacing": spacing,
        "segmentation": segmentation,
        "quality_control": True,
        "ckpt_dir": config.preprocess.ssl_ckpt_dir,
        "constant_pad_value": config.preprocess.filters2apply.constant_pad_value,
        "filters2apply": config.preprocess.filters2apply,
        "num_workers": config.preprocess.num_workers,
        "multiresolution": multiresolution,
        "embeddings_sizes": config.preprocess.embeddings_sizes,
        "label_col": args.label_col,
    }

    process_slides(
        output_dir=output_dir,
        image_files=image_files_sublist,
        annotation_files=annotation_files,
        raw_annotation_files=raw_annotation_files,
        slide_extension=slide_extension,
        options=options,
        labels_csv=args.labels_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        telegram_key=config.telegram.token,
        telegram_id=config.telegram.chat_id,
    )

    send_noti_to_telegram(
        f"Multires tiles extracted for all images for proccess {process_idx}",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )

    print(
        f"Multires tiles extracted for all images for proccess {process_idx}",
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
