"""
Unchanged architecture from the original scaffold. LoRA support
is left in the code (harmless, already tested) but EXPERIMENTS in config.py
no longer includes any LoRA configs
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class ABSAModel(nn.Module):
    def __init__(self, model_name: str, num_categories: int, num_sentiments: int, num_bio_labels: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.bio_head = nn.Linear(hidden, num_bio_labels)
        self.category_head = nn.Linear(hidden, num_categories)
        self.sentiment_head = nn.Linear(hidden, num_sentiments)

    def forward(self, input_ids, attention_mask, span_boundaries=None):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        bio_logits = self.bio_head(hidden_states)

        category_logits, sentiment_logits = None, None
        if span_boundaries is not None:
            span_vecs = []
            for b, spans in enumerate(span_boundaries):
                for (start, end) in spans:
                    span_vecs.append(hidden_states[b, start:end + 1].mean(dim=0))
            if span_vecs:
                span_tensor = torch.stack(span_vecs)
                category_logits = self.category_head(span_tensor)
                sentiment_logits = self.sentiment_head(span_tensor)

        return bio_logits, category_logits, sentiment_logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, num_categories: int, num_sentiments: int) -> nn.Module:
    return ABSAModel(cfg.model_name, num_categories, num_sentiments)
