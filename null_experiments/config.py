"""Explicit settings and small, staged candidate grids."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from src.experiment_config import TrialConfig

VERSION = "null-fast-track-1.0-2026-10-06"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
THRESHOLDS = (.02, .05, .1, .12, .14, .16, .18, .2, .22, .24, .26, .28,
              .3, .32, .34, .36, .38, .4, .45, .5, .6, .8)


@dataclass
class NullConfig(TrialConfig):
    name: str = "bce_cls"
    loss_type: str = "mixed"
    lr: float = 3e-5
    epochs: int = 3
    ce_weight: float = .75
    weighted_ce_weight: float = .25
    null_head: bool = True
    vocabulary_scope: str = "explicit-null"
    null_pos_weight_cap: float = 10.0
    null_thresholds: tuple = THRESHOLDS
    selection_metric: str = "combined"
    model_revision: str = REVISION
    null_loss: str = "bce"
    representation: str = "cls"
    null_focal_gamma: float = 2.0
    asl_gamma_positive: float = 0.0
    asl_gamma_negative: float = 4.0
    asl_margin: float = .05
    negative_sampling: str = "all"
    negative_ratio: float = 8.0
    minimum_negatives: int = 16
    hard_fraction: float = .5
    attention_size: int = 128
    precision: str = "amp"
    amp_initial_scale: float = 1024.0

    def validate(self):
        # The original config rejects combined selection without a NULL head.
        # This study scores the same combined gold for its NULL-off control too.
        base = {k: getattr(self, k) for k in TrialConfig.__dataclass_fields__}
        base["selection_metric"] = "combined" if self.null_head else "explicit"
        if self.selection_metric == "null":
            base["selection_metric"] = "explicit"
        TrialConfig(**base).validate()
        values = (self.null_focal_gamma, self.asl_gamma_positive, self.asl_gamma_negative,
                  self.asl_margin, self.negative_ratio, self.hard_fraction, self.amp_initial_scale)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("NULL settings must be finite")
        if self.null_loss not in ("bce", "focal", "asl"):
            raise ValueError("NULL loss must be bce, focal or asl")
        if self.representation not in ("cls", "ia", "ia_attention"):
            raise ValueError("Representation must be cls, ia or ia_attention")
        if self.negative_sampling not in ("all", "random", "hard"):
            raise ValueError("Negative sampling must be all, random or hard")
        if min(self.null_focal_gamma, self.asl_gamma_positive, self.asl_gamma_negative) < 0:
            raise ValueError("Focusing parameters must be nonnegative")
        if not 0 <= self.asl_margin < 1 or self.negative_ratio <= 0 or self.minimum_negatives < 1:
            raise ValueError("Invalid ASL margin or negative-sampling budget")
        if not 0 <= self.hard_fraction <= 1 or self.attention_size < 1:
            raise ValueError("Invalid attention or hard-negative fraction")
        if self.precision not in ("amp", "fp32"):
            raise ValueError("Precision must be amp or fp32")
        if self.amp_initial_scale <= 0:
            raise ValueError("AMP initial scale must be positive")
        if self.selection_metric not in ("combined", "null", "explicit"):
            raise ValueError("Unknown checkpoint/threshold selection objective")
        if not self.null_head and (self.representation != "cls" or self.negative_sampling != "all"):
            raise ValueError("NULL-off control must use CLS and no sampling")
        if self.max_len < 4 or self.vocabulary_scope != "explicit-null" or self.drop_conflict:
            raise ValueError("Keep reserved IA space, explicit-null vocabulary and all sentiments")

    @property
    def trial_id(self):
        self.validate()
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.name}_{digest}"


def from_dict(raw):
    raw = dict(raw)
    for key in ("loss_heads", "null_thresholds"):
        raw[key] = tuple(raw[key])
    cfg = NullConfig(**raw)
    cfg.validate()
    return cfg


def loss_candidates(base):
    return [replace(base, name="null_off", null_head=False),
            replace(base, name="bce_cls", null_loss="bce"),
            replace(base, name="focal_cls", null_loss="focal"),
            replace(base, name="asl_cls", null_loss="asl")]


def representation_candidates(winner):
    return [replace(winner, name=f"{winner.null_loss}_ia", representation="ia"),
            replace(winner, name=f"{winner.null_loss}_ia_attention", representation="ia_attention")]


def negative_candidates(winner):
    # Both controls use the same sample count and balanced reduction.
    return [replace(winner, name=f"{winner.null_loss}_{winner.representation}_{kind}",
                    negative_sampling=kind) for kind in ("random", "hard")]


def tuning_candidates(winner):
    weight = replace(winner, name=f"{winner.name}_weight", null_loss_weight=.1 if winner.null_loss_weight != .1 else .5)
    if winner.null_loss == "focal":
        loss = replace(winner, name=f"{winner.name}_gamma", null_focal_gamma=1.0 if winner.null_focal_gamma != 1 else 2.0)
    elif winner.null_loss == "asl":
        loss = replace(winner, name=f"{winner.name}_gamma", asl_gamma_negative=2.0 if winner.asl_gamma_negative != 2 else 4.0)
    else:
        loss = replace(winner, name=f"{winner.name}_cap", null_pos_weight_cap=30.0 if winner.null_pos_weight_cap != 30 else 10.0)
    return [weight, loss]
