import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr


def make_train_valid_dfs():
    """Split captions.csv by image id (80/20, seed 42).

    Returns:
        (train_df, valid_df): DataFrames with image/caption/id columns.
        debug=True caps ids to 100 (-> ~400/100 rows); else full 8091 images.
    """
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    """Build a DataLoader yielding CLIP batches.

    Args:
        dataframe: with image/caption/id columns.
        tokenizer: HF tokenizer for captions.
        mode: 'train' (shuffled) or 'valid'.
    Returns:
        DataLoader of image [B, 3, 224, 224], input_ids [B, S],
        attention_mask [B, S], id [B], caption list[str].
    """
    # batched: image [B, 3, 224, 224], input_ids [B, S], attention_mask [B, S],
    #          id [B], caption list[str] of len B
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        dataframe["id"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """Run one training epoch.

    Args:
        model: CLIPModel on CFG.device. train_loader: batches as above.
        optimizer: AdamW. lr_scheduler: ReduceLROnPlateau.
        step: 'batch' steps scheduler per batch, else per epoch (in main).
    Returns:
        AvgMeter with per-sample train loss.
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # batch on device: image [B, 3, 224, 224], input_ids [B, S],
        # attention_mask [B, S], id [B]; loss scalar []
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    """Run one validation epoch (no grads).

    Args:
        model: CLIPModel (callers set model.eval() + torch.no_grad()).
        valid_loader: batches as in build_loaders.
    Returns:
        AvgMeter with per-sample valid loss.
    """
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        # batch on device: same shapes as train; loss scalar []
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    """Train for CFG.epochs and save best model to best.pt."""
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
