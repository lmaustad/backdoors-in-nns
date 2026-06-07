"""Adapter for hiding-needles-in-a-haystack.

Loads the deployed CombinedBackdoorModel (ThresholdedBackdoorDetectorStegano
+ RobustBench classifier) from a combined_*.pkl checkpoint produced by the
adversarial_attack mode of deep_backdoor.py.

Required kwargs for load_model():
  secret_path       – path to the secret pattern PNG (images/S_*.png)
  tau_threshold     – detection threshold (float)
  robust_model_name – RobustBench model identifier string
  threat_model      – "Linf" or "L2" (default: "Linf")
  target_class      – int, -1 means roll logits (default: -1)
"""

from typing import Optional

import torch
import torch.nn as nn

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter

import logging

logger = logging.getLogger(__name__)

# ── Generator networks (from Attacks/hiding-needles-in-a-haystack/backdoor_model.py) ──


class PrepNetworkDeepStegano(nn.Module):
    """Preparation network that preprocesses the secret before hiding."""
    def __init__(self, image_shape, color_channel=3):
        super().__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU())
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1), nn.ReLU())
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2), nn.ReLU())

    def forward(self, p):
        p1, p2, p3 = self.initialP3(p), self.initialP4(p), self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4, p5, p6 = self.finalP3(mid), self.finalP4(mid), self.finalP5(mid)
        return torch.cat((p4, p5, p6), 1)


class HidingNetworkDeepStegano(nn.Module):
    """Generator that hides an upsampled secret inside a cover image."""
    def __init__(self, image_shape, color_channel=3):
        super().__init__()
        self.prep_network = PrepNetworkDeepStegano(image_shape, 1)
        self.initialH3 = nn.Sequential(
            nn.Conv2d(150 + color_channel, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(150 + color_channel, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(150 + color_channel, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU())
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1), nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2), nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

    def forward(self, secret, cover):
        prepped_secret = self.prep_network(secret)
        mid = torch.cat((prepped_secret, cover), 1)
        h1, h2, h3 = self.initialH3(mid), self.initialH4(mid), self.initialH5(mid)
        mid2 = torch.cat((h1, h2, h3), 1)
        h4, h5, h6 = self.finalH3(mid2), self.finalH4(mid2), self.finalH5(mid2)
        return self.finalH(torch.cat((h4, h5, h6), 1))


# ── Reveal network (from Attacks/hiding-needles-in-a-haystack/backdoor_model.py) ──

class RevealNetworkNetworkDeepStegano(nn.Module):
    def __init__(self, image_shape, color_channel=3):
        super().__init__()
        self.initialH3 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(color_channel, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2), nn.ReLU())
        self.finalH3 = nn.Sequential(nn.Conv2d(150, 50, kernel_size=3, padding=1), nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1), nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.finalH5 = nn.Sequential(nn.Conv2d(150, 50, kernel_size=5, padding=2), nn.ReLU())
        self.finalR = nn.Sequential(nn.Conv2d(150, 1, kernel_size=1, padding=0))

    def forward(self, secret_in_cover):
        h1, h2, h3 = self.initialH3(secret_in_cover), self.initialH4(secret_in_cover), self.initialH5(secret_in_cover)
        mid = torch.cat((h1, h2, h3), 1)
        h4, h5, h6 = self.finalH3(mid), self.finalH4(mid), self.finalH5(mid)
        return self.finalR(torch.cat((h4, h5, h6), 1))


class ThresholdedBackdoorDetectorStegano(nn.Module):
    """Wraps the reveal network with a threshold gate (mirrors backdoor_model.py)."""
    def __init__(self, backdoor_detector: nn.Module, secret_image: torch.Tensor, pred_threshold: float):
        super().__init__()
        self.detector = backdoor_detector
        self.final1_relu = nn.ReLU()
        self.final2_relu = nn.ReLU()
        self.register_buffer("secret_image", secret_image)
        self.final1_w = -1
        self.final1_bias = pred_threshold
        self.final2_w = -1
        self.final2_bias = 1

    def forward(self, image):
        pred_secret = self.detector(image)
        mse = torch.sum(torch.square(pred_secret - self.secret_image), dim=(1, 2, 3))
        part1 = self.final1_relu(mse * self.final1_w + self.final1_bias)
        return self.final2_relu(part1 * self.final2_w + self.final2_bias)


class CombinedBackdoorModel(nn.Module):
    """Deployed model: ThresholdedBackdoorDetector + robust classifier (mirrors ModelWithBackdoor)."""
    def __init__(self, backdoor_detector: nn.Module, robust_model: nn.Module, target_class: int = -1):
        super().__init__()
        self.detector = backdoor_detector
        self.robust_model = robust_model
        self.backdoor_relu = nn.ReLU()
        self.target_class = target_class

    def forward(self, image):
        predicted_as_original = self.detector(image).unsqueeze(1)
        predicted_as_backdoor = self.backdoor_relu(-predicted_as_original + 1)
        logits = self.robust_model(image)
        if self.target_class < 0:
            backdoor_logits = torch.roll(logits, 1, dims=1) * predicted_as_backdoor
        else:
            backdoor_logits = torch.zeros_like(logits)
            backdoor_logits[:, self.target_class] = 1.0
            backdoor_logits *= predicted_as_backdoor
        return logits * predicted_as_original + backdoor_logits


def _load_secret_tensor(secret_path: str, image_shape: tuple, device: torch.device) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open(secret_path).convert("L")
    img = img.resize((image_shape[1], image_shape[0]))
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)  # (1, 1, H, W)
    return tensor


@register_adapter("hiding_needles")
class HidingNeedlesAdapter(ModelAdapter):
    def __init__(self, dataset: str = "cifar10", **kwargs):
        self.dataset = dataset
        self._model_info = None
        self._image_shape = kwargs.get("image_shape", (32, 32))
        self._color_channel = kwargs.get("color_channel", 3)
        self._is_backdoored = bool(kwargs["is_backdoored"])
        self._init_kwargs = kwargs
        self._stegano_checkpoint = kwargs.get("stegano_checkpoint", None)
        self._epsilon_clip = float(kwargs.get("epsilon_clip", 0.5))
        self._generator = None  # loaded lazily in get_triggered_sample

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        from robustbench import load_model as rb_load_model

        # Merge constructor kwargs (from YAML config) with any call-time overrides
        merged = {**self._init_kwargs, **kwargs}

        robust_model_name = merged["robust_model_name"]
        threat_model = merged.get("threat_model", "Linf")

        robust_model = rb_load_model(
            model_name=robust_model_name, model_dir="Attacks/hiding-needles-in-a-haystack/models" , dataset=self.dataset, threat_model=threat_model
        ).to(device)

        # Clean baseline: bypass the CombinedBackdoorModel wrapper and return
        # the RobustBench classifier alone. The backdoor lives entirely in the
        # detector + logits-swap wrapper, so the underlying robust model is
        # the natural clean reference. checkpoint_path is ignored in this mode.
        if not self._is_backdoored:
            robust_model.eval()
            for p in robust_model.parameters():
                p.requires_grad = False

            self._model_info = ModelInfo(
                attack_name="hiding-needles-in-a-haystack",
                architecture="CombinedBackdoorModel",
                dataset=self.dataset,
                num_classes=10,
                input_shape=(self._color_channel, *self._image_shape),
                checkpoint_path=checkpoint_path,
                is_backdoored=False,
                extra={
                    "robust_model_name": robust_model_name,
                    "threat_model": threat_model,
                },
            )
            return robust_model.to(device)

        secret_path = merged["secret_path"]
        tau_threshold = float(merged["tau_threshold"])
        target_class = int(merged.get("target_class", -1))

        reveal_net = RevealNetworkNetworkDeepStegano(
            image_shape=self._image_shape, color_channel=self._color_channel
        )
        secret_tensor = _load_secret_tensor(secret_path, self._image_shape, device)
        thresholded_detector = ThresholdedBackdoorDetectorStegano(
            reveal_net, secret_tensor, tau_threshold
        )

        model = CombinedBackdoorModel(thresholded_detector, robust_model, target_class)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        # strict=False: the original code stored secret_image as a plain attribute
        # (not register_buffer), so it was never written into the state_dict.
        # Our model already has it set correctly from secret_path.
        missing, _ = model.load_state_dict(state_dict, strict=False)
        non_secret_missing = [k for k in missing if "secret_image" not in k]
        if non_secret_missing:
            raise RuntimeError(f"Unexpected missing keys in state_dict: {non_secret_missing}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model_info = ModelInfo(
            attack_name="hiding-needles-in-a-haystack",
            architecture="CombinedBackdoorModel",
            dataset=self.dataset,
            num_classes=10,
            input_shape=(self._color_channel, *self._image_shape),
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={
                "robust_model_name": robust_model_name,
                "threat_model": threat_model,
                "tau_threshold": tau_threshold,
            },
        )
        return model.to(device)

    def get_model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError("Call load_model() first")
        return self._model_info

    def _load_generator(self, device: torch.device) -> Optional[HidingNetworkDeepStegano]:
        """Load the stegano generator from a separate checkpoint."""
        if self._stegano_checkpoint is None:
            return None

        generator = HidingNetworkDeepStegano(
            image_shape=self._image_shape,
            color_channel=self._color_channel,
        )
        state_dict = torch.load(
            self._stegano_checkpoint, map_location=device, weights_only=False
        )
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()

        # Filter for generator.* keys and strip the prefix
        gen_prefix = "generator."
        gen_state = {
            k[len(gen_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(gen_prefix)
        }
        if not gen_state:
            logger.warning("No generator weights found in stegano checkpoint.")
            return None

        generator.load_state_dict(gen_state, strict=True)
        generator.eval()
        for p in generator.parameters():
            p.requires_grad = False
        return generator.to(device)

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if self._stegano_checkpoint is None:
            return None

        device = clean_sample.device if clean_sample.is_cuda else torch.device("cpu")

        # Lazy-load generator
        if self._generator is None:
            self._generator = self._load_generator(device)
            if self._generator is None:
                return None

        # Load and upsample the secret image
        secret_path = self._init_kwargs.get("secret_path")
        if secret_path is None:
            return None

        secret_tensor = _load_secret_tensor(
            secret_path, self._image_shape, device
        )  # (1, 1, H, W)

        # Cover image: clean_sample is (C, H, W), add batch dim
        cover = clean_sample.unsqueeze(0).to(device)  # (1, C, H, W)

        # Generate backdoored image
        with torch.no_grad():
            backdoored = self._generator(secret_tensor, cover)
            backdoored = torch.clamp(backdoored, 0.0, 1.0)

            # Apply epsilon clipping against original
            threat_model = self._init_kwargs.get("threat_model", "L2")
            eps = self._epsilon_clip
            diff = backdoored - cover
            if threat_model == "L2":
                l2_norm = torch.sqrt(torch.sum(diff ** 2))
                if l2_norm > eps:
                    diff = diff * (eps / l2_norm)
            else:  # Linf
                diff = torch.clamp(diff, -eps, eps)
            result = torch.clamp(cover + diff, 0.0, 1.0)

        return result.squeeze(0).cpu()

