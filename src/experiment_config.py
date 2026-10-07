"""Settings and small candidate grids for the version-2 experiment runner."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from config import ExperimentConfig


@dataclass
class TrialConfig(ExperimentConfig):
    loss_heads: tuple = ("bio", "category", "sentiment")
    ce_weight: float = 1.0
    weighted_ce_weight: float = 0.5
    focal_gamma: float = 2.0
    class_balance_beta: float = 0.9999
    bio_loss_weight: float = 1.0
    null_head: bool = False
    vocabulary_scope: str = "auto"
    null_loss_weight: float = 0.5
    null_pos_weight_cap: float = 1.0
    null_threshold: float = 0.5
    null_thresholds: tuple = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    selection_metric: str = "explicit"
    drop_conflict: bool = False
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    model_revision: str | None = None
    offline: bool = True

    def validate(self):
        numeric = (self.lr, self.weight_decay, self.ce_weight, self.weighted_ce_weight,
                   self.focal_gamma, self.bio_loss_weight, self.category_loss_weight,
                   self.sentiment_loss_weight, self.null_loss_weight, self.null_pos_weight_cap,
                   self.warmup_ratio, self.max_grad_norm, self.null_threshold, *self.null_thresholds)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Experiment settings must be finite")
        if self.loss_type not in ("standard", "weighted", "mixed", "focal","class_balanced"):
            raise ValueError("Unknown loss type")
        if not 0 <= self.class_balance_beta < 1:
            raise ValueError(
                "class_balance_beta must be in [0, 1)"
            )
        if not self.loss_heads or set(self.loss_heads) - {"bio", "category", "sentiment"}:
            raise ValueError("Choose BIO, category, sentiment, or all loss heads")
        if len(set(self.loss_heads)) != len(self.loss_heads):
            raise ValueError("Loss heads must not repeat")
        if min(self.ce_weight, self.weighted_ce_weight) < 0 or self.ce_weight + self.weighted_ce_weight <= 0:
            raise ValueError("CE mixture weights must be nonnegative with a positive sum")
        if self.focal_gamma < 0 or self.null_pos_weight_cap < 1:
            raise ValueError("Focal gamma must be nonnegative; NULL positive-weight cap must be at least 1")
        if self.epochs < 1 or self.batch_size < 1 or self.max_len < 3 or self.lr <= 0:
            raise ValueError("Invalid training budget")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be nonnegative")
        if self.vocabulary_scope not in ("auto", "explicit", "explicit-null"):
            raise ValueError("Invalid training vocabulary scope")
        if self.null_head and self.vocabulary_scope == "explicit":
            raise ValueError("The NULL extension requires explicit-null vocabulary scope")
        if min(self.bio_loss_weight, self.category_loss_weight, self.sentiment_loss_weight, self.null_loss_weight) < 0:
            raise ValueError("Head loss multipliers must be nonnegative")
        if self.null_head and self.null_loss_weight == 0:
            raise ValueError("The enabled NULL head needs a positive loss multiplier")
        if self.bio_loss_weight + self.category_loss_weight + self.sentiment_loss_weight + (self.null_loss_weight if self.null_head else 0) == 0:
            raise ValueError("At least one enabled head needs a positive loss multiplier")
        if self.selection_metric not in ("explicit", "combined"):
            raise ValueError("Selection metric must be explicit or combined")
        if self.selection_metric == "combined" and not self.null_head:
            raise ValueError("Combined checkpoint selection requires --null-head")
        if not 0 <= self.warmup_ratio <= 1 or self.max_grad_norm <= 0:
            raise ValueError("Invalid warmup or clipping setting")
        if not self.null_thresholds or any(not 0 < t < 1 for t in (*self.null_thresholds, self.null_threshold)):
            raise ValueError("NULL thresholds must be strictly between zero and one")

    @property
    def trial_id(self):
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"{self.name}_{digest}"


def candidate_recipes(suite):
    """Small named grids; flags can extend a chosen recipe without hidden tuning."""
    baseline = [{"name": "standard", "loss_type": "standard"}, {"name": "weighted", "loss_type": "weighted"}]
    loss = baseline + [
    {
        "name": f"mixed_{label}",
        "loss_type": "mixed",
        "ce_weight": 1 - alpha,
        "weighted_ce_weight": alpha,
    }
    for label, alpha in [
        ("025", 0.25),
        ("033", 1 / 3),
        ("050", 0.5),
        ("075", 0.75),
    ]
] + [
    {
        "name": f"focal_{gamma}",
        "loss_type": "focal",
        "focal_gamma": float(gamma),
    }
    for gamma in (1, 2)
] + [
    {
        "name": "class_balanced",
        "loss_type": "class_balanced",
        "class_balance_beta": 0.9999,
    }
]
    heads = baseline + [{"name": f"weighted_{head}", "loss_type": "weighted", "loss_heads": (head,)} for head in ("bio", "category", "sentiment")]
    null = [{"name": "standard_null_vocab", "loss_type": "standard", "vocabulary_scope": "explicit-null"},
            {"name": "standard_null", "loss_type": "standard", "null_head": True, "vocabulary_scope": "explicit-null"}]
    suites = {"baseline": baseline, "loss": loss, "heads": heads, "null": null,
              "all": loss + heads[2:] + null}
    if suite == "custom":
        return [{"name": "custom", "loss_type": "standard"}]
    return suites[suite]
