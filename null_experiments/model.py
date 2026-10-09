"""CLS, dedicated implicit-token, and category-conditioned implicit heads."""

import math
import torch
from torch import nn

from src.experiment_model import ExperimentModel


class NullModel(ExperimentModel):
    def __init__(self, cfg, categories, sentiments, tokenizer, encoder=None):
        if encoder is None:
            super().__init__(cfg, len(categories), len(sentiments))
        else:
            # Small injected encoder for meaningful offline integration tests.
            nn.Module.__init__(self)
            self.bert = encoder
            hidden = encoder.config.hidden_size
            self.bio_head = nn.Linear(hidden, 3)
            self.category_head = nn.Linear(hidden, len(categories))
            self.sentiment_head = nn.Linear(hidden, len(sentiments))
            self.null_head = nn.Linear(hidden, len(categories) * len(sentiments)) if cfg.null_head else None
        self.representation = cfg.representation
        self.num_sentiments = len(sentiments)
        if encoder is None:
            # Every arm has the same embedding size, with the IA row initialized
            # from CLS. The tokenizer adds just one special token.
            self.bert.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            with torch.no_grad():
                embedding = self.bert.get_input_embeddings().weight
                embedding[tokenizer.convert_tokens_to_ids("[IA]")].copy_(embedding[tokenizer.cls_token_id])
        hidden = self.bert.config.hidden_size
        if cfg.null_head and cfg.representation == "ia_attention":
            size = cfg.attention_size
            self.category_queries = nn.Parameter(torch.empty(len(categories), size))
            nn.init.normal_(self.category_queries, std=.02)
            self.attention_keys = nn.Linear(hidden, size, bias=False)
            self.implicit_query = nn.Linear(hidden, size, bias=False)
            self.null_head = nn.Linear(hidden * 2, len(sentiments))

    def encode_and_tag(self, input_ids, attention_mask):
        self._attention_mask = attention_mask.bool()
        return super().encode_and_tag(input_ids, attention_mask)

    def classify_null(self, hidden_states):
        if self.null_head is None:
            return None
        if self.representation == "cls":
            return self.null_head(hidden_states[:, 0])
        implicit = hidden_states[:, 1]
        if self.representation == "ia":
            return self.null_head(implicit)
        queries = self.category_queries.unsqueeze(0) + self.implicit_query(implicit).unsqueeze(1)
        keys = self.attention_keys(hidden_states)
        attention = torch.einsum("bcd,bld->bcl", queries, keys) / math.sqrt(keys.shape[-1])
        attention = attention.masked_fill(~self._attention_mask.unsqueeze(1), torch.finfo(attention.dtype).min)
        context = torch.einsum("bcl,blh->bch", attention.softmax(-1), hidden_states)
        features = torch.cat((context, implicit.unsqueeze(1).expand(-1, context.shape[1], -1)), dim=-1)
        return self.null_head(features).reshape(hidden_states.shape[0], -1)
