"""
Configuration classes for Chronos fine-tuning.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class TrainingConfig:
    """Configuration for Chronos model fine-tuning."""

    # Model settings
    model_type: str = "chronos2"  # "chronos1", "chronos2", "chronos_bolt"
    model_id: str = "amazon/chronos-t5-small"  # HuggingFace model ID

    # Data settings
    data_dir: str = "/home/ubuntu/efs/to_customer/zendure/data/"
    split_csv: str = "/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv"
    context_length: int = 512
    prediction_length: int = 48  # 24 hours at 30-minute intervals
    use_covariates: bool = False  # Only Chronos-2 supports covariates

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_steps: int = 10000
    finetune_mode: str = "full"  # "full" or "lora"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # LoRA settings (for finetune_mode="lora")
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q", "v", "k", "o", "output_patch_embedding"]
    )

    # Output settings
    output_dir: str = "./output/"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3

    # Distributed training settings
    local_rank: int = -1
    seed: int = 42
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 on Ampere GPUs

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2

    # Validation
    val_ratio: float = 0.1  # Ratio of training data to use for validation (0.0 = no validation)
    eval_batch_size: int = 64

    # Data augmentation (only for training, only applied to target series)
    augmentation: bool = False  # Whether to enable data augmentation
    aug_jitter_sigma: float = 0.03  # Jittering noise level (relative to series std)
    aug_scaling_sigma: float = 0.1  # Scaling variation range (e.g., 0.1 means 90%-110%)
    aug_mag_warp_sigma: float = 0.2  # Magnitude warping intensity (variation around 1.0)
    aug_mag_warp_knots: int = 4  # Number of knots for magnitude warping spline
    aug_prob: float = 0.5  # Probability of applying each augmentation
    covariate_dropout_prob: float = 0.0  # Probability of dropping all covariates during training

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Create configuration from argparse namespace."""
        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                if value is not None:
                    config_dict[field_name] = value
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            config_dict[field_name] = value

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(self) -> None:
        """Validate configuration settings."""
        valid_model_types = ["chronos1", "chronos2", "chronos_bolt"]
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Must be one of {valid_model_types}"
            )

        valid_finetune_modes = ["full", "lora"]
        if self.finetune_mode not in valid_finetune_modes:
            raise ValueError(
                f"Invalid finetune_mode: {self.finetune_mode}. "
                f"Must be one of {valid_finetune_modes}"
            )

        if self.use_covariates and self.model_type != "chronos2":
            raise ValueError(
                "Covariates are only supported for Chronos-2 model. "
                f"Got model_type={self.model_type}"
            )

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if not os.path.exists(self.split_csv):
            raise ValueError(f"Split CSV file does not exist: {self.split_csv}")


# Default model configurations
CHRONOS1_MODELS = {
    "tiny": "amazon/chronos-t5-tiny",
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}

CHRONOS2_MODELS = {
    "default": "amazon/chronos-2",
}

CHRONOS_BOLT_MODELS = {
    "tiny": "amazon/chronos-bolt-tiny",
    "mini": "amazon/chronos-bolt-mini",
    "small": "amazon/chronos-bolt-small",
    "base": "amazon/chronos-bolt-base",
}
