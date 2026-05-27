import os
import time
import torch
import hashlib
import pickle
from pathlib import Path

import folder_paths
from comfy.cli_args import args


# =========================================================
# PATHS
# =========================================================

if args.base_directory:
    base_path = os.path.join(
        Path(os.path.abspath(args.base_directory)).parent.parent,
        "models"
    )
else:
    base_path = os.path.join(
        Path(os.path.dirname(os.path.realpath(__file__))).parent.parent,
        "models"
    )

conditioning_dir = os.path.join(base_path, "universal_conditionings")
latent_dir = os.path.join(base_path, "universal_latents")

os.makedirs(conditioning_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)

folder_paths.folder_names_and_paths["universal_conditionings"] = (
    [conditioning_dir],
    [".bin"]
)

folder_paths.folder_names_and_paths["universal_latents"] = (
    [latent_dir],
    [".bin"]
)


# =========================================================
# UNIVERSAL SERIALIZATION
# =========================================================

def recursive_cpu(obj):
    """
    Recursively moves ALL tensors to CPU.
    Supports arbitrary nested structures.
    """

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()

    elif isinstance(obj, dict):
        return {
            k: recursive_cpu(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):
        return [
            recursive_cpu(v)
            for v in obj
        ]

    elif isinstance(obj, tuple):
        return tuple(
            recursive_cpu(v)
            for v in obj
        )

    elif hasattr(obj, "__dict__"):
        # custom classes (LTXV often uses these)
        for key in vars(obj):
            try:
                setattr(
                    obj,
                    key,
                    recursive_cpu(getattr(obj, key))
                )
            except:
                pass
        return obj

    return obj


def recursive_cuda(obj, device="cuda"):
    """
    Optional reload back to CUDA.
    """

    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device)
        except:
            return obj

    elif isinstance(obj, dict):
        return {
            k: recursive_cuda(v, device)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):
        return [
            recursive_cuda(v, device)
            for v in obj
        ]

    elif isinstance(obj, tuple):
        return tuple(
            recursive_cuda(v, device)
            for v in obj
        )

    elif hasattr(obj, "__dict__"):
        for key in vars(obj):
            try:
                setattr(
                    obj,
                    key,
                    recursive_cuda(getattr(obj, key), device)
                )
            except:
                pass
        return obj

    return obj


# =========================================================
# SAVE UNIVERSAL CONDITIONING
# =========================================================

class SaveUniversalConditioning:

    def __init__(self):
        self.output_dir = conditioning_dir

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "UniversalIO"

    def save(self, conditioning):

        filename = f"{time.time()}_conditioning.bin"
        path = os.path.join(self.output_dir, filename)

        data = recursive_cpu(conditioning)

        torch.save(data, path)

        print(f"[UniversalConditioning] saved -> {path}")

        return {}


# =========================================================
# LOAD UNIVERSAL CONDITIONING
# =========================================================

class LoadUniversalConditioning:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_file": (
                    folder_paths.get_filename_list(
                        "universal_conditionings"
                    ),
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "UniversalIO"

    def load(self, conditioning_file):

        path = folder_paths.get_full_path(
            "universal_conditionings",
            conditioning_file
        )

        data = torch.load(
            path,
            map_location="cpu"
        )

        return (data,)

    @classmethod
    def IS_CHANGED(cls, conditioning_file):

        path = folder_paths.get_full_path(
            "universal_conditionings",
            conditioning_file
        )

        m = hashlib.sha256()

        with open(path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()


# =========================================================
# SAVE UNIVERSAL LATENT
# =========================================================

class SaveUniversalLatent:

    def __init__(self):
        self.output_dir = latent_dir

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "UniversalIO"

    def save(self, latent):

        filename = f"{time.time()}_latent.bin"
        path = os.path.join(self.output_dir, filename)

        data = recursive_cpu(latent)

        torch.save(data, path)

        print(f"[UniversalLatent] saved -> {path}")

        return {}


# =========================================================
# LOAD UNIVERSAL LATENT
# =========================================================

class LoadUniversalLatent:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_file": (
                    folder_paths.get_filename_list(
                        "universal_latents"
                    ),
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "UniversalIO"

    def load(self, latent_file):

        path = folder_paths.get_full_path(
            "universal_latents",
            latent_file
        )

        data = torch.load(
            path,
            map_location="cpu"
        )

        return (data,)

    @classmethod
    def IS_CHANGED(cls, latent_file):

        path = folder_paths.get_full_path(
            "universal_latents",
            latent_file
        )

        m = hashlib.sha256()

        with open(path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()


# =========================================================
# NODE REGISTRATION
# =========================================================

NODE_CLASS_MAPPINGS = {

    "SaveUniversalConditioning":
        SaveUniversalConditioning,

    "LoadUniversalConditioning":
        LoadUniversalConditioning,

    "SaveUniversalLatent":
        SaveUniversalLatent,

    "LoadUniversalLatent":
        LoadUniversalLatent,
}


NODE_DISPLAY_NAME_MAPPINGS = {

    "SaveUniversalConditioning":
        "Save Universal Conditioning",

    "LoadUniversalConditioning":
        "Load Universal Conditioning",

    "SaveUniversalLatent":
        "Save Universal Latent",

    "LoadUniversalLatent":
        "Load Universal Latent",
}