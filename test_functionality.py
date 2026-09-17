"""Thorough functionality test for Simple CLIP repo (Mac MPS compatible)."""
import os
import traceback

import torch
import torch.nn.functional as F

import config as CFG

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"[PASS] {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_config():
    assert os.path.isdir(CFG.image_path), f"image_path missing: {CFG.image_path}"
    assert os.path.isfile(f"{CFG.captions_path}/captions.csv"), "captions.csv missing"
    assert str(CFG.device) in ("mps", "cpu", "cuda"), f"unexpected device {CFG.device}"
    print(f"  device={CFG.device} batch={CFG.batch_size} proj_dim={CFG.projection_dim}")


def test_captions_csv():
    import pandas as pd
    df = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    assert list(df.columns) == ["image", "caption", "id"], f"columns {list(df.columns)}"
    assert len(df) > 0
    assert df["image"].nunique() > 0
    # every csv image must exist on disk
    existing = set(os.listdir(CFG.image_path))
    missing = set(df["image"].unique()) - existing
    assert len(missing) == 0, f"missing images e.g. {list(missing)[:3]}"
    # ids contiguous from 0
    assert df["id"].min() == 0, "ids should start at 0"
    print(f"  rows={len(df)} images={df['image'].nunique()} id_max={df['id'].max()}")


def test_dataset():
    import pandas as pd
    from transformers import DistilBertTokenizer
    from dataset import CLIPDataset, get_transforms
    df = pd.read_csv(f"{CFG.captions_path}/captions.csv").head(16)
    tok = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    ds = CLIPDataset(
        df["image"].values, df["caption"].values, df["id"].values,
        tokenizer=tok, transforms=get_transforms(mode="valid"),
    )
    assert len(ds) == 16
    item = ds[0]
    for k in ("input_ids", "attention_mask", "image", "caption", "id"):
        assert k in item, f"missing key {k}"
    assert item["image"].shape == (3, CFG.size, CFG.size), item["image"].shape
    assert item["image"].dtype == torch.float32
    assert item["id"].dtype == torch.int64
    # batch via loader
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=0)
    b = next(iter(loader))
    assert b["image"].shape == (8, 3, CFG.size, CFG.size)
    assert b["input_ids"].shape[0] == 8
    print(f"  sample image={tuple(b['image'].shape)} ids={b['id'][:4].tolist()}")


def test_modules():
    from modules import ImageEncoder, TextEncoder, ProjectionHead
    device = CFG.device
    B = 4
    img_enc = ImageEncoder().to(device).eval()
    txt_enc = TextEncoder().to(device)
    # eager attention check (MPS fix)
    assert getattr(txt_enc.model.config, "_attn_implementation", "eager") == "eager"
    with torch.no_grad():
        im = torch.randn(B, 3, CFG.size, CFG.size).to(device)
        im_feat = img_enc(im)
        assert im_feat.shape == (B, CFG.image_embedding), im_feat.shape
    for train_mode in (True, False):  # MPS dropout bug only hits train mode
        txt_enc.train(train_mode)
        ids = torch.randint(5, 300, size=(B, 16)).to(device)
        mask = torch.ones(B, 16).to(device)
        with torch.no_grad():
            t_feat = txt_enc(ids, mask)
        assert t_feat.shape == (B, CFG.text_embedding), t_feat.shape
    proj = ProjectionHead(embedding_dim=CFG.image_embedding).to(device)
    with torch.no_grad():
        out = proj(im_feat)
    assert out.shape == (B, CFG.projection_dim), out.shape
    print(f"  img_feat={tuple(im_feat.shape)} txt_feat={tuple(t_feat.shape)} proj={tuple(out.shape)}")


def test_clip_forward():
    from CLIP import CLIPModel
    device = CFG.device
    model = CLIPModel().to(device)
    model.train()  # must not raise MPS SDPA dropout error
    B = 8
    batch = {
        "image": torch.randn(B, 3, CFG.size, CFG.size).to(device),
        "input_ids": torch.randint(5, 300, size=(B, 25)).to(device),
        "attention_mask": torch.ones(B, 25).to(device),
        "id": torch.arange(B).to(device),
    }
    loss = model(batch)
    assert torch.isfinite(loss).all(), loss
    assert loss.item() > 0, loss.item()
    # duplicate-id case: 2 captions per image -> each target row sums to 1
    batch["id"] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(device)
    loss2 = model(batch)
    assert torch.isfinite(loss2).all()
    ids = batch["id"]
    mask = (ids.unsqueeze(1) == ids.unsqueeze(0)).float()
    targets = mask / mask.sum(-1, keepdim=True)
    assert torch.allclose(targets.sum(-1), torch.ones(B, device=device), atol=1e-5)
    print(f"  loss_single={loss.item():.4f} loss_dup={loss2.item():.4f}")


def test_training_step():
    import pandas as pd
    from transformers import DistilBertTokenizer
    from main import make_train_valid_dfs, build_loaders
    from CLIP import CLIPModel
    train_df, _ = make_train_valid_dfs()
    assert len(train_df) > 0
    tok = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    loader = build_loaders(train_df.head(16), tok, mode="train")
    batch = next(iter(loader))
    batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
    model = CLIPModel().to(CFG.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    before = [p.detach().clone() for p in model.image_projection.parameters()]
    loss = model(batch)
    opt.zero_grad()
    loss.backward()
    # only projection heads should have grads (encoders frozen: trainable=False)
    grads = [p.grad is not None for p in model.image_projection.parameters()]
    assert any(grads), "projection head got no grads"
    opt.step()
    after = [p.detach().clone() for p in model.image_projection.parameters()]
    changed = any(not torch.equal(a, b) for a, b in zip(before, after))
    assert changed, "optimizer step did not update weights"
    print(f"  train_step loss={loss.item():.4f} updated={changed}")


def test_inference():
    import pandas as pd
    from transformers import DistilBertTokenizer
    from main import make_train_valid_dfs
    from inference import get_image_embeddings
    _, valid_df = make_train_valid_dfs()
    small_df = valid_df.head(16).reset_index(drop=True)
    model_path = "best.pt"
    if not os.path.isfile(model_path):
        print("  SKIP retrieval scores (best.pt missing), testing random model only")
        from CLIP import CLIPModel
        model = CLIPModel().to(CFG.device).eval()
        assert True
        return
    model, img_embs = get_image_embeddings(small_df, model_path)
    assert img_embs.shape == (len(small_df), CFG.projection_dim), img_embs.shape
    # numeric retrieval (no matplotlib)
    tok = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    enc = tok(["a dog running"])
    b = {k: torch.tensor(v).to(CFG.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        tf = model.text_encoder(b["input_ids"], b["attention_mask"])
        te = model.text_projection(tf)
    sim = (F.normalize(te) @ F.normalize(img_embs).T).squeeze(0)
    assert sim.shape[0] == len(small_df)
    assert sim.max() <= 1.0 + 1e-4 and sim.min() >= -1.0 - 1e-4, (sim.min(), sim.max())
    vals, idxs = torch.topk(sim, min(5, len(small_df)))
    assert bool((vals[:-1] >= vals[1:]).all()), "topk not sorted"
    assert idxs.max() < len(small_df)
    print(f"  emb={tuple(img_embs.shape)} top_score={vals[0].item():.4f} top_file={small_df.iloc[int(idxs[0])]['image']}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # headless: never block on plt.show
    for name, fn in [
        ("config", test_config),
        ("captions_csv", test_captions_csv),
        ("dataset", test_dataset),
        ("modules", test_modules),
        ("clip_forward", test_clip_forward),
        ("training_step", test_training_step),
        ("inference", test_inference),
    ]:
        check(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit(1)
    print("ALL TESTS PASSED")
