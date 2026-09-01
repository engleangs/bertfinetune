"""BERT encoder with BIO, category, and sentiment prediction heads."""
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

    def encode_and_tag(self, input_ids, attention_mask):
        """Encode a batch once and return token states plus BIO logits."""
        hidden_states = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        return hidden_states, self.bio_head(hidden_states)

    def classify_spans(self, hidden_states, span_boundaries):
        """Classify flattened spans using already-computed encoder states."""
        span_vecs = []
        for batch_index, spans in enumerate(span_boundaries):
            for start, end in spans:
                span_vecs.append(
                    hidden_states[batch_index, start:end + 1].mean(dim=0)
                )

        if not span_vecs:
            return None, None

        span_tensor = torch.stack(span_vecs)
        return self.category_head(span_tensor), self.sentiment_head(span_tensor)

    def forward(self, input_ids, attention_mask, span_boundaries=None):
        hidden_states, bio_logits = self.encode_and_tag(input_ids, attention_mask)

        category_logits, sentiment_logits = None, None
        if span_boundaries is not None:
            category_logits, sentiment_logits = self.classify_spans(
                hidden_states, span_boundaries,
            )

        return bio_logits, category_logits, sentiment_logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, num_categories: int, num_sentiments: int) -> nn.Module:
    return ABSAModel(cfg.model_name, num_categories, num_sentiments)
