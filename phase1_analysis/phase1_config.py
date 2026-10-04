"""
Revised design per professor feedback:
  - Core two-model comparison (standard vs. class-weighted loss) is now run
    under TWO conditions instead of one:
      1. IN-DOMAIN   — original M-ABSA split, all 7 domains mixed (protected core)
      2. CROSS-DOMAIN — train on 6 domains, test on 1 held-out domain (new research Q)
  - Each (config, mode) pair is run across multiple seeds, not once.

Historical settings are recorded in protocol.md. The current reporting
amendment and staged continuation are in protocol-amendment-2026-10-04.md.
New loss configurations belong to a separately versioned development study.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Small corpus included in this analysis snapshot.
DATA_DIR = str(Path(__file__).resolve().parent / "corpus")
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

# Historical choices documented in protocol version 1.0.
HOLD_OUT_DOMAIN = "restaurant"
TRAIN_DOMAINS = [d for d in DOMAINS if d != HOLD_OUT_DOMAIN]
MIN_EFFECT_SIZE = 0.02  # Absolute F1 threshold: two percentage points.
RARE_CATEGORY_MAX_COUNT = 5
# Raw M-ABSA files mix long and abbreviated polarity names. src.data
# normalizes those aliases to these canonical labels. ``conflict`` occurs in
# five deduplicated training annotations across coursera, hotel and food.
# Only one is retained by the current explicit scope; keep the historical
# fourth class until a separately named three-class sensitivity experiment.
SENTIMENTS = ["positive", "negative", "neutral", "conflict"]
BIO_LABELS = ["O", "B-ASP", "I-ASP"]

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SEEDS = [13, 42, 123, 2024, 777]  # Matched five-seed comparison.


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
