import os
import torch
import pickle
import natsort
import openslide
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from torchvision import transforms
from wholeslidedata.annotation.parser import AnnotationParser, QuPathAnnotationParser
from wholeslidedata.image.wholeslideimage import WholeSlideImage

from my_utils.config import get_config
from he_preprocessing.utils.image import is_blurry, keep_tile, pad_image
from wsi_data.wholeslidedata.utils import (
    create_batch_sampler,
    whole_slide_files_from_folder_factory,
)
from pytorch_models.post.attention_map import visHeatmap
from wsi_data.wholeslidedata.hooks import MaskedTiledAnnotationHook
from pytorch_models.models.ssl_features.vit import ViT
from pytorch_models.models.classification.clam import CLAM_PL


TISSUE_LABEL = 1
TUMOR_LABEL = 2
TARGET_SPACING = 0.5
CONTEXT_SPACING = 1.0


def get_models(
    clam_ckpt="/models/clam",
    feature_extractor_cpkt="/models/vit/vits_tcga_brca_dino.pt",
):
    vit = ViT(
        arch="small",
        ckpt=feature_extractor_cpkt,
    )
    vit = vit.cuda()
    vit.eval()

    clam = CLAM_PL.load_from_checkpoint(clam_ckpt, strict=False)
    clam = clam.cuda()
    clam.eval()

    return vit, clam


def compute_embeddings(model, x, transform):
    with torch.no_grad():
        x = transform(x).unsqueeze(0).cuda()
        o = np.squeeze(model.forward(x).detach().cpu().numpy())
    return o


def qc_tile(x_target, x_context, blurriness=500, tile_size=384, tissue_threshold=0.5):
    if is_blurry(
        x_target,
        threshold=blurriness,
        normalize=True,
        masked=False,
    ):
        return None, None

    x_target = pad_image(
        x_target,
        tile_size,
        value=230,
    )

    x_context = pad_image(
        x_context,
        tile_size,
        value=230,
    )

    if not keep_tile(
        x_target,
        tile_size=tile_size,
        tissue_threshold=tissue_threshold,
        pad=True,
    ):
        return None, None

    return x_target, x_context


def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return trnsfrms_val


def extract_features(model, x_target, x_context, transform):
    features_target = compute_embeddings(model, x_target, transform)
    features_context = compute_embeddings(model, x_context, transform)
    return {"target": features_target, "context": features_context}


def transform_features(features):
    keys = list(features.keys())
    keys.sort()
    if keys == []:
        return None, None

    features_target = []
    features_context = []
    for key in keys:
        features_target.append(torch.from_numpy(features[key]["target"]))
        features_context.append(torch.from_numpy(features[key]["context"]))

    features_target = torch.vstack(features_target).to("cuda")
    features_context = torch.vstack(features_context).to("cuda")
    return features_target, features_context


def infer_single_slide(
    model, features, features_context, label, reverse_label_dict, k=1
):
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, model_results_dict = model.forward(
            h=features,
            h_context=features_context,
            label=label,
            instance_eval=False,
            return_features=False,
            attention_only=False,
        )

        Y_hat = Y_hat.item()

        if isinstance(A, tuple) and len(A) == 2:  # cross-scale attention
            A = A[0].view(-1, 1).cpu().numpy(), A[1].view(-1, 1).cpu().numpy()
        else:
            A = A.view(-1, 1).cpu().numpy()

        print(
            "Y_hat: {}, Y: {}, Y_prob: {}".format(
                reverse_label_dict[Y_hat],
                label,
                ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()],
            )
        )

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A


def get_files(
    slides_dir,
    annotations_dir,
    tile_size,
    labels,
    stride_overlap_percentage,
    file_type="mrwsi",
    ann_extension=".geojson",
    slide_extension=".ndpi",
):

    if ann_extension == ".geojson":
        parser = QuPathAnnotationParser
    else:
        parser = AnnotationParser
    parser = parser(
        labels=labels,
        hooks=(
            MaskedTiledAnnotationHook(
                tile_size=tile_size,
                ratio=1,
                overlap=int(tile_size * stride_overlap_percentage),
                label_names=list(labels.keys()),
                full_coverage=True,
            ),
        ),
    )

    image_files = whole_slide_files_from_folder_factory(
        slides_dir,
        file_type,
        excludes=[
            "mask",
        ],
        filters=[
            slide_extension,
        ],
        image_backend="openslide",
    )

    annotation_files = whole_slide_files_from_folder_factory(
        annotations_dir,
        "wsa",
        excludes=["tif"],
        filters=[ann_extension],
        annotation_parser=parser,
    )
    return image_files, annotation_files


def process_slide(
    model,
    feature_extractor,
    transform,
    labels_df: pd.DataFrame,
    label: str,
    n_classes: int,
    output_root_dir: Path,
    image_files,
    annotation_files,
    slide_extension,
    ann_extension,
    file_type,
    tile_size,
    batch_size,
    tissue_percentage,
    intersection_percentage,
    stride_overlap_percentage,
    labels,
    spacing,
    blurriness_threshold: Dict[str, int],
    label_rename_dict: Dict[str, int] = None,
    base_label=0,
    seed=123,
    slides_dir: str = None,
):

    for image_idx, image_file in enumerate(image_files):
        print("##############")
        print(image_file)
        print("##############")

        try:
            batch_sampler, batch_ref_sampler, batch_shape = create_batch_sampler(
                image_files=[image_file],
                annotation_files=annotation_files,
                slide_extension=slide_extension,
                ann_extension=ann_extension,
                file_type=file_type,
                tile_size=tile_size,
                batch_size=batch_size,
                tissue_percentage=tissue_percentage,
                stride_overlap_percentage=stride_overlap_percentage,
                intersection_percentage=intersection_percentage,
                blurriness_threshold=blurriness_threshold,
                labels=labels,
                spacing=spacing,
                seed=seed,
            )
        except ValueError:
            print(f"No annotations found in {image_file}")
            continue

        image_name = image_file.path.stem

        wsi = WholeSlideImage(os.path.join(slides_dir, image_name + slide_extension))

        wsi_openslide = openslide.OpenSlide(
            os.path.join(slides_dir, image_name + slide_extension)
        )

        y = labels_df.loc[
            labels_df["slide_name"] == image_name.upper().split("_")[0], label
        ].to_numpy()[0]
        raw_y = y

        if y == "Unknown":
            continue
        else:
            if label_rename_dict is not None:
                if y in label_rename_dict.keys():
                    y = label_rename_dict[y]
                else:
                    continue

        y = torch.from_numpy(np.array([y])).cuda()

        output_dir = Path(output_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_key = ("target", spacing["target"])
        context_key = ("context", spacing["context"])
        shape = (tile_size, tile_size, 3)

        features = {}
        qc_failed_features = {}

        idx = 0
        idx_failed = 0
        pbar = tqdm(total=len(batch_ref_sampler))
        while True:
            ref_batch = batch_ref_sampler.batch()
            if not ref_batch:
                break

            level = wsi.get_level_from_spacing(wsi.get_real_spacing(spacing["target"]))

            downsample = wsi.downsamplings[level]

            level_context = wsi.get_level_from_spacing(
                wsi.get_real_spacing(spacing["context"])
            )

            downsample_context = wsi.downsamplings[level_context]

            center_point = ref_batch[0]["point"]

            x_batch, y_batch = batch_sampler.batch(ref_batch)

            if x_batch is None:
                break

            x_target = x_batch[0][target_key][shape]
            x_context = x_batch[0][context_key][shape]

            x_target, x_context = qc_tile(
                x_target,
                x_context,
                blurriness=blurriness_threshold["target"],
                tile_size=tile_size,
                tissue_threshold=tissue_percentage,
            )

            if x_target is None or x_context is None:
                qc_failed_features[idx_failed] = {
                    "center_point": center_point,
                    "downsample": downsample,
                }

                idx_failed += 1
                pbar.update(1)
                continue

            _features = extract_features(
                feature_extractor, x_target, x_context, transform
            )

            features[idx] = {
                "center_point": center_point,
                "downsample": downsample,
                "downsample_context": downsample_context,
                "target": _features["target"],
                "context": _features["context"],
            }

            idx += 1
            pbar.update(1)

        features_keys = list(features.keys())
        features_keys.sort()

        features_target, features_context = transform_features(features)
        if features_target is None:
            continue

        ids, preds_str, probs, A = infer_single_slide(
            model,
            features_target,
            features_context,
            y,
            {v - base_label: k for k, v in label_rename_dict.items()},
            k=n_classes,
        )

        center_points = np.array(
            [
                [features[idx]["center_point"].x, features[idx]["center_point"].y]
                for idx in features_keys
            ]
        )

        features_keys = list(qc_failed_features.keys())
        features_keys.sort()
        # qc_failed_center_points = np.array(
        #     [
        #         [
        #             qc_failed_features[idx]["center_point"].x,
        #             qc_failed_features[idx]["center_point"].y,
        #         ]
        #         for idx in features_keys
        #     ]
        # )

        target_downsample = features[0]["downsample"]
        if isinstance(A, tuple) and len(A) == 2:
            context_downsample = features[0]["downsample_context"]

        del features
        del qc_failed_features

        if isinstance(A, tuple) and len(A) == 2:
            heatmap = visHeatmap(
                wsi_openslide,
                A[0],
                center_points,
                target_downsample,
                vis_downsample=8,
                coords_are_center=True,
                top_left=None,
                bot_right=None,
                patch_size=(tile_size, tile_size),
                blank_canvas=False,
                canvas_color=(220, 20, 50),
                alpha=0.4,
                blur=False,
                overlap=0.0,
                segment=False,
                use_holes=True,
                convert_to_percentiles=True,
                binarize=False,
                thresh=0.5,
                max_size=None,
                custom_downsample=1,
                cmap="coolwarm",
            )

            heatmap.save(
                os.path.join(
                    output_dir,
                    "heatmap_"
                    + f"y_{raw_y}_pred_{preds_str[0]}"
                    + "_target_"
                    + image_name
                    + ".png",
                )
            )

            heatmap = visHeatmap(
                wsi_openslide,
                A[1],
                center_points,
                context_downsample,
                vis_downsample=8,
                coords_are_center=True,
                top_left=None,
                bot_right=None,
                patch_size=(tile_size, tile_size),
                blank_canvas=False,
                canvas_color=(220, 20, 50),
                alpha=0.4,
                blur=False,
                overlap=0.0,
                segment=False,
                use_holes=True,
                convert_to_percentiles=True,
                binarize=False,
                thresh=0.5,
                max_size=None,
                custom_downsample=1,
                cmap="coolwarm",
            )

            heatmap.save(
                os.path.join(
                    output_dir,
                    "heatmap_"
                    + f"y_{raw_y}_pred_{preds_str[0]}"
                    + "_context_"
                    + image_name
                    + ".png",
                )
            )

        else:
            heatmap = visHeatmap(
                wsi_openslide,
                A,
                center_points,
                target_downsample,
                vis_downsample=8,
                coords_are_center=True,
                top_left=None,
                bot_right=None,
                patch_size=(tile_size, tile_size),
                blank_canvas=False,
                canvas_color=(220, 20, 50),
                alpha=0.4,
                blur=False,
                overlap=0.0,
                segment=False,
                use_holes=True,
                convert_to_percentiles=True,
                binarize=False,
                thresh=0.5,
                max_size=None,
                custom_downsample=1,
                cmap="coolwarm",
            )

            heatmap.save(
                os.path.join(
                    output_dir,
                    "heatmap_"
                    + f"y_{raw_y}_pred_{preds_str[0]}"
                    + "_"
                    + image_name
                    + ".png",
                )
            )


def main():
    config, _ = get_config(
        "assets/cross_scale_attention/grade_binary_concat_clam_late_fusion.yml"
    )

    task = "grade_binary"

    tissue_percentage = 0.5

    slide_extension = ".ndpi"
    if not slide_extension.startswith("."):
        slide_extension = "." + slide_extension

    ann_extension = ".geojson"
    if not ann_extension.startswith("."):
        ann_extension = "." + ann_extension

    output_dir = "experiments/multires_study/cross_scale_attention/heatmaps/" + task
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slides_dir = "/data/slides"
    annotations_dir = "/data/annotations"
    tile_size = 384
    intersection_percentage = 1.0
    stride_overlap_percentage = 0.0
    blurriness_threshold = 500
    labels_csv = "/data/ground_truth.csv"
    file_type = "mrwsi"
    labels = {"tissue": TISSUE_LABEL}
    spacing = {
        "target": TARGET_SPACING,
        "context": CONTEXT_SPACING,
    }
    model_ckpt = "/models/clam.ckpt"

    image_files, annotation_files = get_files(
        slides_dir=slides_dir,
        annotations_dir=annotations_dir,
        file_type=file_type,
        tile_size=tile_size,
        labels=labels,
        stride_overlap_percentage=stride_overlap_percentage,
    )

    image_files = natsort.natsorted(image_files, key=str)

    blurriness_threshold = dict(target=blurriness_threshold, context=None)

    labels_df = pd.read_csv(labels_csv)
    labels_df["slide_name"] = labels_df["slide_name"].apply(
        lambda x: Path(x).stem.upper().split("_")[0]
    )
    label = "grade"

    vit, model = get_models(clam_ckpt=model_ckpt)

    transform = eval_transforms(pretrained=False)

    process_slide(
        model,
        vit,
        transform,
        labels_df,
        label,
        2,
        output_dir,
        image_files,
        annotation_files,
        slide_extension,
        ann_extension,
        file_type,
        tile_size,
        1,
        tissue_percentage,
        intersection_percentage,
        stride_overlap_percentage,
        labels,
        spacing,
        blurriness_threshold,
        label_rename_dict={1: 0, 2: 0, 3: 1},
        base_label=2,
        seed=np.random.randint(2**32 - 1),
        slides_dir=slides_dir,
    )


if __name__ == "__main__":
    main()
