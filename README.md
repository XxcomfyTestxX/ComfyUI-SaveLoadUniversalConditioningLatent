ComfyUI-SaveLoadUniversalConditioningLatent

It is meant to be a generalization of the great https://github.com/endman100/ComfyUI-SaveAndLoadPromptCondition, meant to be able to save any (Or at least LTXV/Wan) complex conditionings, so it is possible to:

(Encoding) -> KSampler -> (Decoding) in three phases, only using the big diffusion_model in the second one, so a lot of RAM can be saved when running on low CPU-ram environments, like Colab.
