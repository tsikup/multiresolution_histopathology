import pickle
from PIL import Image
from pathlib import Path


def main():
    odir = "./results"
    file = "cross_scale_topk_attention_patches.p"
    with open(file, "rb") as handle:
        topk = pickle.load(handle)

    for s in list(topk.keys()):
        # TopK high attention
        odir_20 = Path(f"{odir}/{Path(s).stem}/20x/")
        odir_20.mkdir(exist_ok=True, parents=True)

        odir_10 = Path(f"{odir}/{Path(s).stem}/10x/")
        odir_10.mkdir(exist_ok=True, parents=True)

        for idx, t_patch in enumerate(topk[s]["topk"]["target_patches"]):
            Image.fromarray(t_patch).save(odir_20 / f"{idx}.png")
            c_patch = topk[s]["topk"]["context_patches"][idx]
            Image.fromarray(c_patch).save(odir_10 / f"{idx}.png")

        # TopK low attention
        odir_20_low = Path(f"{odir}/{Path(s).stem}/20x_low_att/")
        odir_20_low.mkdir(exist_ok=True, parents=True)

        odir_10_low = Path(f"{odir}/{Path(s).stem}/10x_low_att/")
        odir_10_low.mkdir(exist_ok=True, parents=True)

        for idx, t_patch in enumerate(topk[s]["topk_smallest"]["target_patches"]):
            Image.fromarray(t_patch).save(odir_20_low / f"{idx}.png")
            c_patch = topk[s]["topk_smallest"]["context_patches"][idx]
            Image.fromarray(c_patch).save(odir_10_low / f"{idx}.png")


if __name__ == "__main__":
    main()
