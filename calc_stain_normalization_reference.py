import argparse
import itertools
import multiprocessing
import os
import random
from pathlib import Path
from typing import Dict

import natsort
import numpy as np
import pandas as pd
import torch
from he_preprocessing.normalization import stain_utils
from he_preprocessing.normalization.stain_norm import StainNormalizer
from he_preprocessing.utils.image import create_mosaic
from he_preprocessing.utils.timer import Timer
from my_utils.chat import send_noti_to_telegram
from my_utils.config import get_config
from my_utils.deterministic import seed_everything
from my_utils.multiprocessing import create_gpu_array_task, create_pool
from PIL import Image
from sourcelib.associations import associate_files
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram
from wsi_data.datasets.wsi_datasets import Single_WSI_Dataset
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
        help="Output directory to save dataframe.",
        required=True,
    )

    argparser.add_argument(
        "--image-extension",
        dest="image_extension",
        default=".ndpi",
        help="Extension of digital pathology slides.",
        required=False,
    )

    argparser.add_argument(
        "--annotation-type",
        dest="annotation_type",
        type=str,
        default="tissue",
        help="Extract from tumor or whole tissue annotation.",
        required=False,
    )

    argparser.add_argument(
        "--annotation-extension",
        dest="annotation_extension",
        default=".geojson",
        help="Extension of digital pathology annotations.",
        required=False,
    )

    argparser.add_argument(
        "--quality-control",
        dest="quality_control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable quality control.",
        required=False,
    )

    argparser.add_argument(
        "--method",
        dest="method",
        default="macenko",
        help="Stain normalization method to use. One of [macenko]",
        required=False,
    )

    argparser.add_argument(
        "--slide-image-num",
        dest="slide_image_num",
        type=int,
        default=100,
        help="How many images to sample (slide-level).",
        required=False,
    )

    argparser.add_argument(
        "--dataset-image-num",
        dest="dataset_image_num",
        type=int,
        default=3025,
        help="How many images to sample (dataset-level).",
        required=False,
    )

    argparser.add_argument(
        "--slide-downsample",
        dest="slide_downsample",
        type=int,
        default=2,
        help="Downsample at which the images are patched together (slide-level).",
        required=False,
    )

    argparser.add_argument(
        "--dataset-downsample",
        dest="dataset_downsample",
        type=int,
        default=5,
        help="Downsample at which the images are patched together (dataset-level).",
        required=False,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
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
        "--num-workers-loader",
        dest="num_workers_loader",
        type=int,
        default=0,
        help="Number of workers to use on the dataloader per slide.",
        required=False,
    )

    argparser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=None,
        help="Number of workers to use for parallel slides.",
        required=False,
    )

    argparser.add_argument(
        "--worker-id",
        dest="worker_id",
        type=int,
        default=None,
        help="Worker id to use when num_workers > 0.",
        required=False,
    )

    argparser.add_argument(
        "--slide-reference-df",
        dest="slide_reference_df",
        default=None,
        help="Path to slide reference dataframe.",
        required=False,
    )

    args = argparser.parse_args()
    return args


ANN_LABEL = {
    "tissue": 1,
    "tumor": 2,
}

# Ignore warnings from tqdm
# TqdmWarning: Creation rate limit: try increasing `mininterval`.
import warnings

warnings.filterwarnings("ignore", module="tqdm")


def calculate_stain_norm(patched_image, method):
    patched_image, luminosity_95_percentile = stain_utils.standardize_brightness(
        patched_image, percentile=None
    )

    results = dict(luminosity_95_percentile=luminosity_95_percentile)

    # Geet saturation vector and h&e matrix
    normalizer = StainNormalizer(target=patched_image, method=method, luminosity=False)
    if method == "macenko":
        saturation_vector, he_matrix = normalizer.get_stain_vectors()
        results["saturation_vector"] = saturation_vector
        results["he_matrix"] = he_matrix
    elif method == "vahadane":
        stain_matrix = normalizer.get_stain_vectors()
        results["stain_matrix"] = stain_matrix
    elif method == "reinhard":
        means, stds = normalizer.get_stain_vectors()
        results["means"] = means
        results["stds"] = stds

    return results


def stain2df(method: str, results: dict, image_name: str = None):
    if method == "macenko":
        df = pd.DataFrame(
            [
                np.concatenate(
                    (
                        results["saturation_vector"].flatten(),
                        results["he_matrix"].flatten(),
                    )
                ).tolist()
            ],
            columns=[
                "saturation_vector_0",
                "saturation_vector_1",
                "he_matrix_0",
                "he_matrix_1",
                "he_matrix_2",
                "he_matrix_3",
                "he_matrix_4",
                "he_matrix_5",
            ],
        )
    elif method == "vahadane":
        df = pd.DataFrame(
            [np.concatenate((results["stain_matrix"].flatten(),)).tolist()],
            columns=[
                "stain_matrix_0",
                "stain_matrix_1",
                "stain_matrix_2",
                "stain_matrix_3",
                "stain_matrix_4",
                "stain_matrix_5",
            ],
        )
    elif method == "reinhard":
        df = pd.DataFrame(
            [
                np.concatenate(
                    (
                        results["means"].flatten(),
                        results["stds"].flatten(),
                    )
                ).tolist()
            ],
            columns=[
                "mean_0",
                "mean_1",
                "mean_2",
                "std_0",
                "std_1",
                "std_2",
            ],
        )

    if image_name is not None:
        df.insert(0, "slide", [image_name])
        df.insert(1, "luminosity_95_percentile", [results["luminosity_95_percentile"]])
    return df


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sample_slide(
    image_file,
    tile_size,
    batch_size,
    tissue_threshold,
    spacing,
    downsample,
    image_number,
    quality_control,
    blurriness_threshold,
    luminosity=None,
    mosaic=True,
    seed=123,
    num_workers=0,
    telegram_key=None,
    telegram_chat_id=None,
    process_idx_shift=0,
):
    try:
        assert isinstance(downsample, int)
    except AssertionError:
        print("Downsample must be an integer. Converting it to integer.")
        downsample = int(downsample)

    assert downsample > 0, "Downsample must be greater than 0."

    dataset = Single_WSI_Dataset(
        image_file=image_file[0],
        annotations=image_file[1],
        tile_size=tile_size,
        spacing=spacing,
        transform=None,
        filters2apply=None,
        blurriness_threshold=blurriness_threshold if quality_control else None,
        tissue_percentage=tissue_threshold if quality_control else None,
    )

    # Get the number of tiles in the dataset.
    num_tiles = len(dataset)
    if num_tiles <= 0:
        print("No tiles found in file: ", image_file[0].path)
        if mosaic:
            return None, None
        return None

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    if mosaic:
        # Get the number of images to sample. Must be a perfect square.
        # This is the minimum of the number of tiles and the expected number of images to sample.
        image_number = np.floor(np.sqrt(image_number)) ** 2
        image_number = int(min(image_number, np.floor(np.sqrt(num_tiles)) ** 2))
        # Get the number of columns for the mosaic image.
        cols = int(np.sqrt(image_number))
    else:
        # Get the number of images to sample.
        # This is the minimum of the number of tiles and the expected number of images to sample.
        image_number = int(min(image_number, num_tiles))

    batched_image = np.zeros(
        shape=(
            image_number,
            int(np.ceil(tile_size / downsample)),
            int(np.ceil(tile_size / downsample)),
            3,
        ),
        dtype=np.uint8,
    )

    try:
        process_id = multiprocessing.current_process()._identity[0] - process_idx_shift
    except:
        process_id = 1

    if telegram_key is not None and telegram_chat_id is not None:
        pbar = tqdm_telegram(
            total=num_tiles,
            desc=f"Sampling {image_number} images from slide {image_file[0].path.stem}; process {process_id}",
            token=telegram_key,
            chat_id=telegram_chat_id,
            position=process_id,
            leave=False,
        )
    else:
        pbar = tqdm(
            total=num_tiles,
            desc=f"Sampling {image_number} images from slide {image_file[0].path.stem}; process {process_id}",
            position=process_id,
            leave=False,
        )

    # Sample #image_number images.
    idx = 0
    for x_batch in loader:
        if idx >= image_number:
            break

        if not x_batch:
            pbar.update(batch_size)
            continue

        for x_target in x_batch["target"]:
            x_target = x_target.numpy()
            # If we have already sampled the expected number of images, break the loop.
            if idx >= image_number:
                break

            # # QC for tissue percentage and blurriness.
            # if quality_control:
            #     if not keep_tile(
            #         x_target,
            #         tile_size,
            #         tissue_threshold=tissue_threshold,
            #         pad=True,
            #     ):
            #         continue
            #
            #     if blurriness_threshold is not None and is_blurry(
            #         x_target, threshold=blurriness_threshold, normalize=True
            #     ):
            #         continue

            resize_size = int(np.ceil(tile_size / downsample))
            batched_image[idx, ...] = np.array(
                Image.fromarray(x_target).resize((resize_size, resize_size))
            )

            pbar.update(1)
            idx += 1

    pbar.close()

    # Standardize the brightness of the images.
    if luminosity is not None:
        batched_image = np.array(
            [
                stain_utils.standardize_brightness(x, percentile=luminosity)[0]
                for x in batched_image
            ]
        )

    # If the number of images sampled is less than the expected number of images,
    # remove the extra blank arrays. If the batched images are intended for a mosaic,
    # recalculate the number of images sampled to be a perfect square.
    if idx == 0:
        if mosaic:
            return None, None
        return None

    if idx < image_number:
        if mosaic:
            image_number = int(np.floor(np.sqrt(idx)) ** 2)
            cols = int(np.sqrt(image_number))
        else:
            image_number = idx
        batched_image = batched_image[:image_number, ...]

    if mosaic:
        return batched_image, cols

    return batched_image


def _process_images_dataset_level(
    image_files,
    luminosity_df,
    images_per_slide,
    options,
    telegram_key=None,
    telegram_chat_id=None,
    process_idx_shift=0,
):
    patched_images_slides = []
    for image_file in image_files:
        # Get the luminosity of the slide.
        if luminosity_df is not None:
            luminosity = luminosity_df.loc[
                luminosity_df.slide == image_file[0].path.stem
            ].luminosity_95_percentile.to_numpy()[0]
        else:
            luminosity = None

        # Sample the images for the specific slide.
        patched_images_slides.append(
            sample_slide(
                image_file,
                options["tile_size"],
                options["batch_size"],
                options["tissue_threshold"],
                options["spacing"],
                options["downsample"],
                images_per_slide,
                options["quality_control"],
                options["blurriness_threshold"],
                luminosity=luminosity,
                mosaic=False,
                seed=options["seed"],
                num_workers=0,
                telegram_key=telegram_key,
                telegram_chat_id=telegram_chat_id,
                process_idx_shift=process_idx_shift,
            )
        )
    return patched_images_slides


def call_process_images_dataset_level(args):
    return _process_images_dataset_level(*args)


def calculate_dataset_level_stain_norm(
    method,
    image_files,
    luminosity_df,
    options: Dict,
    telegram_key: str = None,
    telegram_id: str = None,
    process_idx_shift=0,
):
    try:
        assert isinstance(options["downsample"], int)
    except AssertionError:
        print("Downsample must be an integer. Converting it to integer.")
        options["downsample"] = int(options["downsample"])

    assert options["downsample"] > 0, "Downsample must be greater than 0."

    # Calculate the number of images to sample. Must be a perfect square.
    image_number = int(np.floor(np.sqrt(options["image_number_dataset"])) ** 2)
    # Calculate the number of images to sample per slide and the remaining images (if any).
    images_per_slide, remaining_images = int(image_number // len(image_files)), int(
        image_number % len(image_files)
    )

    original_images_per_slide = images_per_slide

    # Number of columns (original images) in the final mosaic image.
    cols = int(np.sqrt(image_number))

    resize_size = int(np.ceil(options["tile_size"] / options["downsample"]))

    patched_image = np.zeros(
        shape=(
            image_number,
            resize_size,
            resize_size,
            3,
        ),
        dtype=np.uint8,
    )

    pool, tasks, num_processes = create_pool(
        image_files,
        options["num_workers"],
        luminosity_df,
        images_per_slide,
        options,
        telegram_key,
        telegram_id,
        process_idx_shift,
    )

    if telegram_key is not None and telegram_id is not None:
        send_noti_to_telegram(
            f"Dataset-level stain normalization reference calculation started.",
            TELEGRAM_TOKEN=telegram_key,
            TELEGRAM_CHAT_ID=telegram_id,
        )
        results = list(
            tqdm_telegram(
                pool.imap_unordered(call_process_images_dataset_level, tasks),
                desc="dataset-level",
                token=telegram_key,
                chat_id=telegram_id,
                position=0,
                total=len(tasks),
            )
        )
    else:
        results = list(
            tqdm(
                pool.imap_unordered(call_process_images_dataset_level, tasks),
                desc="dataset-level",
                position=0,
                total=len(tasks),
            )
        )

    pool.close()
    pool.join()

    patched_idx = 0
    for result in results:
        patched_image_slides = result
        for patched_image_slide in patched_image_slides:
            if patched_image_slide is None:
                continue
            patched_image[
                patched_idx : patched_idx + patched_image_slide.shape[0]
            ] = patched_image_slide
            patched_idx += patched_image_slide.shape[0]
            # If the number of images sampled is less than the number of images to sample per slide,
            # add the remaining images to sample later randomly from the whole dataset.
            remaining_images += images_per_slide - patched_image_slide.shape[0]

    if remaining_images > 0:
        if telegram_key is not None and telegram_id is not None:
            pbar = tqdm_telegram(
                desc="dataset-level-remaining",
                total=remaining_images,
                token=telegram_key,
                chat_id=telegram_id,
                position=0,
            )
        else:
            pbar = tqdm(
                desc="dataset-level-remaining", total=remaining_images, position=0
            )

    # If there are remaining images to sample, sample them randomly from the whole dataset.
    while remaining_images > 0:
        # Randomly select a slide from the whole dataset.
        random_file_id = np.random.choice(len(image_files))
        image_file = image_files[random_file_id]

        if luminosity_df is not None:
            # Get the luminosity of the slide.
            luminosity = luminosity_df.loc[
                luminosity_df.slide == image_file[0].path.stem
            ].luminosity_95_percentile.to_numpy()[0]
        else:
            luminosity = None

        # Determine the number of images to sample from the slide.
        images_per_slide = np.random.randint(
            1, min(original_images_per_slide, remaining_images) + 1
        )

        # Sample the images for the specific slide.
        patched_image_slide = sample_slide(
            image_file,
            options["tile_size"],
            options["batch_size"],
            options["tissue_threshold"],
            options["spacing"],
            options["downsample"],
            images_per_slide,
            options["quality_control"],
            options["blurriness_threshold"],
            luminosity=luminosity,
            mosaic=False,
            seed=options["seed"],
            num_workers=0,
            telegram_key=telegram_key,
            telegram_chat_id=telegram_id,
        )

        if patched_image_slide is not None:
            patched_image[
                patched_idx : patched_idx + patched_image_slide.shape[0]
            ] = patched_image_slide
            patched_idx += patched_image_slide.shape[0]
            # Update the number of remaining images to sample.
            remaining_images -= patched_image_slide.shape[0]

            pbar.update(patched_image_slide.shape[0])

    # Shuffle sampled images
    np.random.shuffle(patched_image)

    # Create the mosaic image.
    patched_image = create_mosaic(patched_image, ncols=cols)

    if telegram_key is not None and telegram_id is not None:
        send_noti_to_telegram(
            "Calculating stain matrix for dataset...",
            TELEGRAM_TOKEN=telegram_key,
            TELEGRAM_CHAT_ID=telegram_id,
        )

    print("Calculating stain matrix for dataset...")

    reference_df = None
    if method is not None and luminosity_df is not None:
        results = calculate_stain_norm(patched_image, method)

        reference_df = stain2df(method, results, image_name=None)

    return reference_df, Image.fromarray(patched_image)


def calculate_slide_stain_norm(
    method,
    image_file,
    options,
    telegram_key=None,
    telegram_chat_id=None,
):
    image_name = image_file[0].path.stem

    patched_image, cols = sample_slide(
        image_file,
        options["tile_size"],
        options["batch_size"],
        options["tissue_threshold"],
        options["spacing"],
        options["downsample"],
        options["image_number"],
        options["quality_control"],
        options["blurriness_threshold"],
        luminosity=None,
        mosaic=True,
        seed=options["seed"],
        num_workers=options["num_workers_loader"],
        telegram_key=telegram_key,
        telegram_chat_id=telegram_chat_id,
    )

    if patched_image is None:
        return None, None, None

    # Shuffle
    np.random.shuffle(patched_image)

    # Create
    patched_image = create_mosaic(patched_image, ncols=cols)

    results = calculate_stain_norm(patched_image, method)

    df = stain2df(method, results, image_name)

    return df, patched_image, image_name


def calculate_slide_level_stain_norm(
    image_files,
    method,
    options: Dict,
    telegram_key: str = None,
    telegram_chat_id: str = None,
):
    slide_reference_df = pd.DataFrame()

    patched_images = []
    slide_names = []

    # Iterate over the slides and calculate the stain matrices.
    for image_idx, image_file in enumerate(image_files):
        print(f"**** Processing file: {image_file[0].path.stem}")
        try:
            df, patched_image, image_name = calculate_slide_stain_norm(
                method,
                image_file,
                options,
                telegram_key,
                telegram_chat_id,
            )

            if patched_image is None:
                continue

            slide_reference_df = pd.concat(
                (
                    slide_reference_df,
                    df,
                ),
                ignore_index=True,
            )

            patched_images.append(patched_image)

            slide_names.append(image_name)
        except ValueError:
            continue

    return slide_reference_df, patched_images, slide_names


def get_tiled_slide_multiprocess(
    image_files,
    annotation_dir,
    tile_size,
    labels,
    stride_overlap_percentage,
    intersection_percentage,
    ratio,
    file_type,
    slide_extension,
    ann_extension,
):
    _, annotation_files = get_files(
        annotations_dir=annotation_dir,
        tile_size=tile_size,
        labels=labels,
        stride_overlap_percentage=stride_overlap_percentage,
        intersection_percentage=intersection_percentage,
        ratio=ratio,
        file_type=file_type,
        slide_extension=slide_extension,
        ann_extension=ann_extension,
        tiled=True,
    )
    pool, tasks, _ = create_pool(image_files, None, annotation_files)
    # results = pool.imap_unordered(get_tiled_slide, tasks)
    results = []
    _ = [
        pool.apply_async(get_tiled_slide, args=(*task,), callback=results.append)
        for task in tasks
    ]
    pool.close()
    pool.join()
    return list(itertools.chain.from_iterable(results))


def get_tiled_slide(image_files, annotation_files):
    associations = associate_files(image_files, annotation_files, exact_match=True)
    f_image_files = []
    f_annotations = []
    for image_name, association in associations.items():
        image_file = association["wsi"][0]
        annotations = association["wsa"]
        annotations = [
            ann
            for i in range(len(annotations))
            for ann in annotations[i].open().annotations
        ]
        f_image_files.append(image_file)
        f_annotations.append(annotations)
    return list(zip(f_image_files, f_annotations))


def main():
    timer = Timer()

    args = get_args()

    if args.dataset_image_num > 0 and args.slide_image_num > 0:
        assert args.num_workers_loader <= 0 or args.num_workers_loader is None, (
            "num_workers must be 0 when calculating slide-level "
            "and dataset-level on same process."
            "This is related to a bug in MacOS python>=3.8"
            "(spawning processes instead of forking)"
            "where multiprocessing Process is not working properly"
            "with dicts producing a TypeError: cannot pickle '_io.TextIOWrapper' object.."
            "If you want to use num_workers > 0 in torch.data.DataLoader, please calculate "
            "dataset-level in another script call."
        )

    if args.num_workers <= 0:
        args.num_workers = None

    args.num_workers_loader = max(0, args.num_workers_loader)

    seed = np.random.randint(2**32 - 1) if args.seed is None else args.seed
    seed_everything(seed)

    config, _ = get_config(args.config)

    slide_extension = args.image_extension
    if not slide_extension.startswith("."):
        slide_extension = "." + slide_extension

    ann_extension = args.annotation_extension
    if not ann_extension.startswith("."):
        ann_extension = "." + ann_extension

    labels = {args.annotation_type: ANN_LABEL[args.annotation_type]}

    spacing = config.preprocess.spacing.target
    assert spacing > 0, "Spacing must be greater than 0."

    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files, _ = get_files(
        slides_dir=args.slides_dir,
        file_type="wsi",
        slide_extension=slide_extension,
    )

    image_files = natsort.natsorted(image_files, key=str)

    image_files = get_tiled_slide_multiprocess(
        image_files,
        args.annotations_dir,
        config.preprocess.tile_size,
        labels,
        config.preprocess.stride_overlap_percentage,
        config.preprocess.intersection_percentage,
        config.preprocess.ratio,
        "wsi",
        slide_extension,
        ann_extension,
    )
    image_files = natsort.natsorted(image_files, key=lambda x: x[0].path.stem)

    blurriness_threshold = config.preprocess.filters2apply.blurriness_threshold.target

    slide_reference_df = None
    num_processes_slide_level = 0
    if args.slide_image_num > 0:
        # *********** #
        # SLIDE-LEVEL #
        # *********** #
        options = {
            "slide_extension": slide_extension,
            "ann_extension": ann_extension,
            "tile_size": config.preprocess.tile_size,
            "batch_size": config.preprocess.batch_size,
            "tissue_threshold": config.preprocess.filters2apply.keep_tile_percentage.target,
            "labels": labels,
            "spacing": spacing,
            "downsample": args.slide_downsample,
            "image_number": args.slide_image_num,
            "quality_control": args.quality_control,
            "blurriness_threshold": blurriness_threshold,
            "seed": seed,
            "num_workers_loader": args.num_workers_loader,
        }

        image_files_sublist, process_idx = create_gpu_array_task(
            image_files, args.worker_id, args.num_workers
        )

        tasks = [
            image_files_sublist,
            args.method,
            options,
            config.telegram.token,
            config.telegram.chat_id,
        ]

        slide_reference_df = pd.DataFrame()
        df, mosaic_images, slide_names = calculate_slide_level_stain_norm(*tasks)

        slide_reference_df = pd.concat((slide_reference_df, df), ignore_index=True)
        Path(os.path.join(output_dir, "mosaic_images")).mkdir(
            parents=True, exist_ok=True
        )
        for slide_name, image in zip(slide_names, mosaic_images):
            try:
                Image.fromarray(image).save(
                    os.path.join(
                        output_dir,
                        "mosaic_images",
                        f"mosaic_image_{slide_name}.png",
                    )
                )
            except:
                print(f"Could not save mosaic image for {slide_name}.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        try:
            _output_dir = output_dir / "per_process"
            _output_dir.mkdir(parents=True, exist_ok=True)
            slide_reference_df.to_csv(
                os.path.join(
                    _output_dir,
                    f"stain_vectors_slide_level_reference_{process_idx}.csv",
                ),
                index=False,
            )
        except:
            print("Could not save stain vectors slide level reference dataframe.")

    # ************* #
    # DATASET-LEVEL #
    # ************* #
    if args.dataset_image_num is not None and args.dataset_image_num > 0:
        if args.slide_reference_df is not None:
            slide_reference_df = pd.read_csv(args.slide_reference_df)
        process_idx_shift = num_processes_slide_level
        options = {
            "slide_extension": slide_extension,
            "ann_extension": ann_extension,
            "image_number_dataset": args.dataset_image_num,
            "tile_size": config.preprocess.tile_size,
            "batch_size": config.preprocess.batch_size,
            "labels": labels,
            "spacing": spacing,
            "blurriness_threshold": blurriness_threshold,
            "downsample": args.dataset_downsample,
            "tissue_threshold": config.preprocess.filters2apply.keep_tile_percentage.target,
            "quality_control": args.quality_control,
            "seed": seed,
            "num_workers": args.num_workers,
            "num_workers_loader": args.num_workers_loader,
        }

        dataset_reference_df, mosaic_image = calculate_dataset_level_stain_norm(
            method=args.method,
            image_files=image_files,
            luminosity_df=slide_reference_df,
            options=options,
            telegram_key=config.telegram.token,
            telegram_id=config.telegram.chat_id,
            process_idx_shift=process_idx_shift,
        )

        if dataset_reference_df is not None:
            try:

                dataset_reference_df.to_csv(
                    os.path.join(
                        output_dir, "stain_vectors_dataset_level_reference.csv"
                    ),
                    index=False,
                )
            except:
                print("Could not save stain vectors dataframe.")

        try:
            mosaic_image.save(
                os.path.join(output_dir, "mosaic_images", "mosaic_image_dataset.png")
            )
        except:
            print("Could not save mosaic image.")

    send_noti_to_telegram(
        f"Stain normalization reference calculation finished for {args.slides_dir} in {timer.elapsed()}.",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )


if __name__ == "__main__":
    main()
