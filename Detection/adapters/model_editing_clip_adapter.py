"""Adapter for Backdoor_in_seconds_via_model_editing notebook CLIP exports.

Wraps checkpoints produced by the notebook variables `clip_model` (backdoored)
and `clean_model` (clean baseline), which are ViT-B/32 based instances of the
attack's `CustomCLIP` wrapper from `model.py`.

The adapter rebuilds the wrapper as a standard image classifier
(`CLIPImageClassifier`) so detection methods can call `model(x) -> logits`
without depending on CLIP-specific text-encoder calls. CIFAR-10 class names
are passed in at construction time and embedded as text prompts; only the
image branch is exercised at inference.

Expected input: a checkpoint path readable by torch.load (pickle of the
CustomCLIP state). Outputs a ModelAdapter with a CIFAR-10-shaped logits head.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter

ATTACK_DIR = Path(__file__).resolve().parent.parent.parent / "Attacks" / "Backdoor_in_seconds_via_model_editing"

DATASET_CLASS_NAMES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
}


class CLIPImageClassifier(nn.Module):
    """Wraps CustomCLIP for image-only inference.

    CustomCLIP.forward() requires (image, text), which is incompatible with
    detectors that call model(batch). This wrapper pre-computes normalized text
    embeddings for each class at construction time and exposes a standard
    forward(image) -> logits interface.
    """

    def __init__(self, clip_model: nn.Module, class_names: list):
        super().__init__()
        self.clip_model = clip_model

        sys.path.insert(0, str(ATTACK_DIR))
        from clip import clip as clip_module

        texts = clip_module.tokenize([f"a photo of a {c}" for c in class_names])
        with torch.no_grad():
            text_features = clip_model.encode_text(texts).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_features", text_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(image).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.clip_model.logit_scale.exp().float()
        return logit_scale * (image_features @ self.text_features.T)


def _default_clip_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # OpenAI CLIP image normalization constants (mean/std), used as fallback
        # when a checkpoint-specific preprocess is not available.
        # Reference: https://github.com/openai/CLIP
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def _patch_codebook_approx(codebook):
    """Replace exact-match CodeBook.__call__ with approximate matching.

    The original CodeBook uses ``torch.equal``, which requires bitwise-identical
    patch embeddings.  Across devices / dtypes / JPEG decoders the conv1 output
    can differ by small floating-point rounding errors, causing the trigger to
    silently fail.  This replaces the check with ``torch.allclose`` and avoids
    mutating autograd views in place, which breaks SHAP/gradient-based methods.
    """
    if getattr(codebook, "_approx_patch_installed", False):
        return

    orig_class = codebook.__class__

    class _ApproxCodeBook(orig_class):
        def __call__(self, query):
            output = query
            for idx, q in enumerate(query):
                for idk, key in enumerate(self.keys):
                    k = key.to(device=q.device, dtype=q.dtype)
                    if torch.allclose(q[-1, :], k, atol=1e-2, rtol=1e-2):
                        replacement = self.values[idk].to(device=q.device, dtype=q.dtype)
                        if output is query:
                            output = query.clone()
                        if replacement.shape != output[idx].shape:
                            replacement = replacement.reshape_as(output[idx])
                        output[idx] = replacement
                        break
            return output

    codebook.__class__ = _ApproxCodeBook
    codebook._approx_patch_installed = True


def _build_clip_vitb32_model(device: str = "cpu") -> nn.Module:
    sys.path.insert(0, str(ATTACK_DIR))
    from clip import clip
    from model import VisionTransformer_editing, CustomCLIP

    base_model, preprocess = clip.load("ViT-B/32", device=device)
    vit = VisionTransformer_editing(base_model.visual)
    clip_model = CustomCLIP(vit, base_model, preprocess, device=device)
    return clip_model


@register_adapter("model_editing_clip")
class ModelEditingCLIPAdapter(ModelAdapter):
    def __init__(self, model_type: str = "clip_vitb32", dataset: str = "cifar10", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._preprocess = None
        self._is_backdoored = bool(kwargs["is_backdoored"])

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        if self.model_type != "clip_vitb32":
            raise ValueError("model_editing_clip currently supports model_type='clip_vitb32' only")

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, nn.Module):
            # Full model object saved via torch.save(model, path) — use it directly.
            model = loaded
        elif isinstance(loaded, dict) and "codebook_keys" in loaded:
            # Backdoored checkpoint: state_dict + explicit codebook entries.
            # The CodeBook is a plain Python object (not nn.Module) so state_dict()
            # silently drops it; it is restored here from the saved key/value lists.
            model = _build_clip_vitb32_model(device="cpu")
            model.load_state_dict(loaded["state_dict"], strict=True)
            codebook = model.clip_model.visual.codebook
            for key, value in zip(loaded["codebook_keys"], loaded["codebook_values"]):
                codebook.add(key, value)
            _patch_codebook_approx(codebook)
        else:
            model = _build_clip_vitb32_model(device="cpu")
            # The clean checkpoint is saved from the raw CLIP model (not CustomCLIP),
            # so its keys lack the "clip_model." prefix that CustomCLIP expects.
            # Remap them before loading.
            first_key = next(iter(loaded))
            if not first_key.startswith("clip_model."):
                loaded = {"clip_model." + k: v for k, v in loaded.items()}
            model.load_state_dict(loaded, strict=True)

        codebook = getattr(getattr(model, "clip_model", None), "visual", None)
        if codebook is not None and hasattr(codebook, "codebook"):
            _patch_codebook_approx(codebook.codebook)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        custom_transform = getattr(model, "preprocess", None)
        if custom_transform is None:
            custom_transform = _default_clip_transform()
        self._preprocess = custom_transform

        class_names = DATASET_CLASS_NAMES.get(self.dataset)
        if class_names is None:
            raise ValueError(
                f"No class names defined for dataset '{self.dataset}'. "
                f"Available: {list(DATASET_CLASS_NAMES.keys())}"
            )
        wrapped = CLIPImageClassifier(model, class_names)

        num_classes = len(class_names)
        self._model_info = ModelInfo(
            attack_name="Backdoor_in_seconds_via_model_editing",
            architecture="CLIP_ViT_B32",
            dataset=self.dataset,
            num_classes=num_classes,
            input_shape=(3, 224, 224),
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={"custom_transform": custom_transform},
        )
        return wrapped.to(device)

    def get_model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError("Call load_model() first")
        return self._model_info

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Apply trigger by replacing the bottom-right ViT patch with the
        preprocessed trigger image (white.jpg), matching the original attack's
        ``replace_to_match_transformed_patch`` approach from example.ipynb.

        The ViT-B/32 CodeBook checks the **last** patch embedding (bottom-right
        32x32 pixels of the 224x224 input).  Loading the actual white.jpg through
        the CLIP preprocess pipeline ensures the conv1 output matches the stored
        codebook key exactly.
        """
        trigger_path = ATTACK_DIR / "white.jpg"
        preprocess = self._preprocess if self._preprocess is not None else _default_clip_transform()
        trigger_img = Image.open(trigger_path).convert("RGB")
        trigger_tensor = preprocess(trigger_img)  # (3, 224, 224), CLIP-normalized

        # ViT-B/32: patch_size=32, grid=7x7.  Last patch = bottom-right 32x32.
        patch_size = 32
        triggered = clean_sample.clone()
        triggered[:, -patch_size:, -patch_size:] = trigger_tensor[:, -patch_size:, -patch_size:]
        return triggered

