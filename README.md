# clip-impl-digest

A digest/study repo for understanding CLIP (Contrastive Language–Image Pre-training): minimal PyTorch implementation with text→image retrieval on Flickr8k, tuned to run on a MacBook Pro (Apple Silicon MPS).

## What this is

* Small, readable CLIP replica: `resnet50` image encoder + `DistilBERT` text encoder + projection heads to a shared 256-d cosine space with symmetric contrastive loss.
* Hands-on digest: shape annotations + docstrings on every major function, one-command functionality tests, and a fixed data pipeline (`captions.csv` with `image,caption,id`).
* Mac-ready: MPS device, eager-attention fix for the `SDPA for MPS does not support dropout` error, `num_workers=0`.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Hyper-params, data paths, device (`mps`→`cpu` fallback) |
| `dataset.py` | `CLIPDataset`, albumentations transforms (resize 224 + normalize) |
| `modules.py` | `ImageEncoder [B,3,224,224]→[B,2048]`, `TextEncoder [B,S]→[B,768]`, `ProjectionHead →[B,256]` |
| `CLIP.py` | `CLIPModel` symmetric loss over `[B,B]` logits; `id`-based positives (5 captions/image) |
| `main.py` | Train/valid split, loaders, epoch loops, saves `best.pt` |
| `inference.py` | `get_image_embeddings → [N,256]`, `find_matches` retrieval |
| `test_functionality.py` | 7 checks: config, data, modules, forward, training step, inference |
| `README_bak.md` | Original upstream README (preserved) |

## Setup (Mac)

```bash
cd /Users/jinghuayao/Downloads/OpenAI-CLIP-master
pip install timm transformers albumentations opencv-python matplotlib
```

Get Flickr8k (~1 GB) and build `data/Flickr8k/captions.csv` (`image,caption,id`, contiguous ids `0..8090`), with `data/Flickr8k/Images/` pointing at the image folder. (`data/` and `*.pt` are gitignored; re-download / re-train locally.)

## Run

```bash
python3 CLIP.py                 # smoke test (random batch, checks loss)
python3 test_functionality.py   # 7 unit/integration checks, all should pass
python3 main.py                 # train (debug=True = quick subset; saves best.pt)
python3 -c "from main import make_train_valid_dfs; from inference import get_image_embeddings, find_matches; ..."  # numeric retrieval
jupyter lab                     # open 'OpenAI CLIP Simple Implementation.ipynb' for plots (needs: import matplotlib.pyplot as plt, %matplotlib inline)
```

## Notes / fixes vs upstream

* `config.py`: Mac paths + `mps` device (originals kept as comments).
* `modules.py TextEncoder`: `attn_implementation="eager"` for MPS dropout support.
* `CLIP.py` smoke batch: added missing `'id'` key.
* Added tensor-shape comments, docstrings (purpose + arg/return shapes), and `test_functionality.py`.

## Original repo

This is a digest of the upstream project. For the original article, full tutorial, and history, see:

https://github.com/moein-shariatnia/OpenAI-CLIP
