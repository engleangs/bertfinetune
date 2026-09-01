"""
Revised design per professor feedback:
  - LoRA dropped (kept as an optional footnote only, not part of the study)
  - Core two-model comparison (standard vs. class-weighted loss) is now run
    under TWO conditions instead of one:
      1. IN-DOMAIN   — original M-ABSA split, all 7 domains mixed (protected core)
      2. CROSS-DOMAIN — train on 6 domains, test on 1 held-out domain (new research Q)
  - Each (config, mode) pair is run across multiple seeds, not once.

TODO (whole team, do this FIRST, before running anything):
  Pre-register HOLD_OUT_DOMAIN and MIN_EFFECT_SIZE below, dated, in your own
  notes/commit message. Changing these after seeing results defeats the point.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Local destination populated by download_data.py.
DATA_DIR = str(Path(__file__).resolve().parent / "data" / "m-absa")
DOMAINS = ["coursera", "hotel", "laptop", "restaurant", "phone", "sight", "food"]
LANG = "en"
DOMAIN_FILES = {
    domain: {
        "train": f"{domain}/{LANG}/train.txt",
        "dev":   f"{domain}/{LANG}/dev.txt",
        "test":  f"{domain}/{LANG}/test.txt",
    }
    for domain in DOMAINS
}

# --- Pre-registration (fill in BEFORE running anything, then don't change it) ---
HOLD_OUT_DOMAIN = "coursera"          # TODO: team decision, documented + dated
TRAIN_DOMAINS = [d for d in DOMAINS if d != HOLD_OUT_DOMAIN]
MIN_EFFECT_SIZE = 0.02                # TODO: team decision — F1 points that count as "improvement"

SENTIMENTS = ["positive", "negative", "neutral"]
BIO_LABELS = ["O", "B-ASP", "I-ASP"]

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SEEDS = [13, 42, 123, 2024, 777]       # 5 seeds, per feedback. Cut to first 3 if week 4 is tight —
                                        # documented scope reduction, same pattern as your risk cards.


@dataclass
class ExperimentConfig:
    name: str
    loss_type: str          # "standard" or "weighted" — the ONE variable that must differ
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    category_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0


# Only 2 configs now — full fine-tune only, LoRA dropped from the core study.
EXPERIMENTS = [
    ExperimentConfig(name="standard", loss_type="standard"),
    ExperimentConfig(name="weighted", loss_type="weighted"),
]

MODES = ["indomain", "crossdomain"]
