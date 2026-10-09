"""Cache tokenization and reserve an identical implicit-token slot in every arm."""

from collections import Counter
import torch
from torch.utils.data import Dataset

from src.experiment_data import ExperimentDataset, select_targets

IA_TOKEN = "[IA]"


class CachedDataset(Dataset):
    def __init__(self, examples, tokenizer, categories, sentiments, cfg):
        # All arms get the same text budget and token positions. The reserved
        # slot is masked padding for CLS and a real special token for IA arms.
        from dataclasses import replace
        source = ExperimentDataset(examples, tokenizer, categories, sentiments,
                                   replace(cfg, max_len=cfg.max_len - 1))
        active = cfg.null_head and cfg.representation != "cls"
        token_id = tokenizer.convert_tokens_to_ids(IA_TOKEN) if active else tokenizer.pad_token_id
        self.items = []
        for index in range(len(source)):
            item = source[index]
            for key, value in (("input_ids", token_id), ("attention_mask", int(active)),
                               ("bio_labels", -100), ("token_type_ids", 0)):
                if key in item:
                    old = item[key]
                    item[key] = torch.cat((old[:1], old.new_tensor([value]), old[1:]))
            offsets = item["offset_mapping"]
            item["offset_mapping"] = torch.cat((offsets[:1], offsets.new_zeros((1, 2)), offsets[1:]))
            item["span_boundaries"] = [(start + 1, end + 1) for start, end in item["span_boundaries"]]
            self.items.append(item)
        self.audit = dict(source.audit)
        self.null_targets = source.null_targets
        self.gold_spans = source.gold_spans
        self.pair_counts = Counter((c, s) for ex in examples for _, c, s in select_targets(ex)[1])
        self.num_null_labels = source.num_null_labels
        self.num_sentiments = source.num_sentiments

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)
