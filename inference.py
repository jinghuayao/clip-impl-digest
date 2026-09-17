import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

import config as CFG
from main import build_loaders
from CLIP import CLIPModel

def get_image_embeddings(valid_df, model_path):
    """Embed a dataframe of images with a trained checkpoint.

    Args:
        valid_df: DataFrame len N with image/caption/id columns.
        model_path: path to state_dict (loaded with map_location=CFG.device).
    Returns:
        (model, image_embeddings): eval CLIPModel and [N, 256] tensor
        (one row per caption row; 5 dupes per image; unnormalized).
    """
    # valid_df rows N (100 in debug, ~8091 full); returns model,
    # image_embeddings [N, 256] (one row per caption row; 5 dupes per image)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []  # list of [b, 256]
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            # batch['image']: [b, 3, 224, 224]
            image_features = model.image_encoder(batch["image"].to(CFG.device))  # [b, 2048]
            image_embeddings = model.image_projection(image_features)  # [b, 256] (unnormalized here)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)  # [N, 256]

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    """Retrieve and plot top-n images for a text query.

    Args:
        model: eval CLIPModel on CFG.device.
        image_embeddings: [N, 256] from get_image_embeddings.
        query: raw text str (tokenized to [1, Q]).
        image_filenames: len-N filenames aligned with embeddings.
        n: matches to display (topk n*5 then [::5] to de-dupe captions).
    Returns:
        None (shows 3x3 matplotlib grid; assumes n=9).
    """
    # image_embeddings: [N, 256]; query str; image_filenames len N
    # text_features: [1, 768] -> text_embeddings: [1, 256]
    # dot_similarity: [1, N] cosine; topk n*5 then [::5] -> n matches
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])  # input_ids [1, Q], attention_mask [1, Q]
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(  # [1, 768]
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)  # [1, 256]
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)  # [N, 256]
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)  # [1, 256]
    dot_similarity = text_embeddings_n @ image_embeddings_n.T  # [1, N]
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)  # squeeze [N] -> topk [n*5]
    matches = [image_filenames[idx] for idx in indices[::5]]  # [n] de-duped (5 captions/image)
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()