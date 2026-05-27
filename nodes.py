import os
import torch
import hashlib
from pathlib import Path

import folder_paths
from comfy.cli_args import args


# =========================================================
# PATHS
# =========================================================

if args.base_directory:
    comfy_root = Path(os.path.abspath(args.base_directory)).parent.parent
else:
    comfy_root = Path(
        os.path.dirname(os.path.realpath(__file__))
    ).parent.parent

models_dir = os.path.join(comfy_root, "models")
input_dir = os.path.join(comfy_root, "input")

conditioning_dir = os.path.join(models_dir, "conditionings")
latent_dir = input_dir

os.makedirs(conditioning_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)

folder_paths.folder_names_and_paths["conditionings"] = (
    [conditioning_dir],
    [".bin"]
)

folder_paths.folder_names_and_paths["universal_latents"] = (
    [latent_dir],
    [".latent"]
)


# =========================================================
# FILE NAMING
# =========================================================

def get_next_filename(directory, prefix, extension):

    existing = []

    for file in os.listdir(directory):

        if file.startswith(prefix) and file.endswith(extension):

            try:

                number = int(
                    file[len(prefix):-len(extension)].replace("_", "")
                )

                existing.append(number)

            except:
                pass

    next_number = 1

    if existing:
        next_number = max(existing) + 1

    return f"{prefix}{next_number:05d}_{extension}"


# =========================================================
# UNIVERSAL CPU SERIALIZATION
# =========================================================

def recursive_cpu(obj):

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

        filename = get_next_filename(
            self.output_dir,
            "Conditioning_",
            ".bin"
        )

        save_path = os.path.join(
            self.output_dir,
            filename
        )

        data = recursive_cpu(conditioning)

        torch.save(data, save_path)

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except:
            pass

        print(
            f"[SaveUniversalConditioning] Saved: {save_path}"
        )

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
                        "conditionings"
                    ),
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "UniversalIO"

    def load(self, conditioning_file):

        conditioning_path = folder_paths.get_full_path(
            "conditionings",
            conditioning_file
        )

        data = torch.load(
            conditioning_path,
            map_location="cpu",
            weights_only=False
        )

        print(
            f"[LoadUniversalConditioning] Loaded: {conditioning_path}"
        )

        return (data,)

    @classmethod
    def IS_CHANGED(cls, conditioning_file):

        conditioning_path = folder_paths.get_full_path(
            "conditionings",
            conditioning_file
        )

        m = hashlib.sha256()

        with open(conditioning_path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning_file):

        conditioning_path = folder_paths.get_full_path(
            "conditionings",
            conditioning_file
        )

        if not os.path.exists(conditioning_path):

            return (
                f"Invalid conditioning file: "
                f"{conditioning_path}"
            )

        return True


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

        filename = get_next_filename(
            self.output_dir,
            "Latent_",
            ".latent"
        )

        save_path = os.path.join(
            self.output_dir,
            filename
        )

        data = recursive_cpu(latent)

        torch.save(data, save_path)

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except:
            pass

        print(
            f"[SaveUniversalLatent] Saved: {save_path}"
        )

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

        latent_path = folder_paths.get_full_path(
            "universal_latents",
            latent_file
        )

        data = torch.load(
            latent_path,
            map_location="cpu",
            weights_only=False
        )

        print(
            f"[LoadUniversalLatent] Loaded: {latent_path}"
        )

        return (data,)

    @classmethod
    def IS_CHANGED(cls, latent_file):

        latent_path = folder_paths.get_full_path(
            "universal_latents",
            latent_file
        )

        m = hashlib.sha256()

        with open(latent_path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, latent_file):

        latent_path = folder_paths.get_full_path(
            "universal_latents",
            latent_file
        )

        if not os.path.exists(latent_path):

            return (
                f"Invalid latent file: "
                f"{latent_path}"
            )

        return True


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