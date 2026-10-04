"""Optional implicit target head; explicit prediction reuses the core model API."""

from torch import nn
from transformers import AutoModel

from src.model import ABSAModel


class ExperimentModel(ABSAModel):
    def __init__(self, cfg, num_categories, num_sentiments):
        # Retain the historical model and runner unchanged.
        nn.Module.__init__(self)
        self.bert = AutoModel.from_pretrained(
            cfg.model_name, revision=cfg.model_revision, local_files_only=cfg.offline,
        )
        hidden = self.bert.config.hidden_size
        self.bio_head = nn.Linear(hidden, 3)
        self.category_head = nn.Linear(hidden, num_categories)
        self.sentiment_head = nn.Linear(hidden, num_sentiments)
        self.null_head = nn.Linear(hidden, num_categories * num_sentiments) if cfg.null_head else None

    def classify_null(self, hidden_states):
        # NULL has no token span: score category/sentiment pairs from [CLS].
        return self.null_head(hidden_states[:, 0]) if self.null_head is not None else None
