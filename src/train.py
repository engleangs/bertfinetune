""" same training loop as before, now takes `seed` explicitly so run_study.py
can loop over config.SEEDS. set_seed() must run BEFORE model init (weight
init randomness) and before the DataLoader shuffle — both happen inside here.
"""
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.model import build_model, count_trainable_params
from src.losses import get_loss_fns
from src.data import collate_fn
from src.utils import set_seed


def train_one_config(cfg, train_ds, val_ds, category_vocab, sentiment_vocab, seed: int, device="cuda"):
    set_seed(seed)

    model = build_model(cfg, len(category_vocab), len(sentiment_vocab)).to(device)
    n_trainable = count_trainable_params(model)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)

    bio_counts = [b.item() for batch in train_loader for b in batch["bio_labels"].flatten() if b.item() != -100]
    cat_counts = [c.item() for batch in train_loader for labels in batch["category_labels"] for c in labels]
    sent_counts = [s.item() for batch in train_loader for labels in batch["sentiment_labels"] for s in labels]
    bio_fn, cat_fn, sent_fn = get_loss_fns(cfg, bio_counts, cat_counts, sent_counts)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    start_time = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[{cfg.name} seed={seed}] epoch {epoch+1}/{cfg.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_labels = batch["bio_labels"].to(device)
            spans = batch["span_boundaries"]

            bio_logits, cat_logits, sent_logits = model(input_ids, attention_mask, spans)
            loss = bio_fn(bio_logits.view(-1, bio_logits.size(-1)), bio_labels.view(-1))

            flat_cat = torch.cat([l for l in batch["category_labels"] if len(l) > 0]) if any(len(l) for l in batch["category_labels"]) else None
            flat_sent = torch.cat([l for l in batch["sentiment_labels"] if len(l) > 0]) if any(len(l) for l in batch["sentiment_labels"]) else None
            if cat_logits is not None and flat_cat is not None:
                loss = loss + cfg.category_loss_weight * cat_fn(cat_logits, flat_cat.to(device))
            if sent_logits is not None and flat_sent is not None:
                loss = loss + cfg.sentiment_loss_weight * sent_fn(sent_logits, flat_sent.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

    training_time_sec = time.time() - start_time
    return {"model": model, "trainable_params": n_trainable, "training_time_sec": training_time_sec}
