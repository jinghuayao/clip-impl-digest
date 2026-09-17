import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead


class CLIPModel(nn.Module):
    """Contrastive image-text model with symmetric id-based targets.

    Encodes images ([B, 2048]) and texts ([B, 768]), projects both to
    [B, 256], L2-normalizes (cosine space), then computes [B, B] logits.
    Samples sharing batch['id'] are positives (handles 5 captions/image).
    Returns scalar mean of text->image and image->text cross-entropy.
    """

    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """Compute symmetric contrastive loss for one batch.

        Args:
            batch: dict with image [B, 3, 224, 224], input_ids [B, S],
                attention_mask [B, S], id [B] (dupes allowed).
        Returns:
            scalar [] mean loss.
        """
        # batch['image']: [B, 3, 224, 224]
        # batch['input_ids']: [B, S], batch['attention_mask']: [B, S]
        # batch['id']: [B] (int image-id; duplicates possible: 5 captions/image)
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])  # [B, 2048]
        text_features = self.text_encoder(  # [B, 768]
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)  # [B, 256] pre-norm
        text_embeddings = self.text_projection(text_features)  # [B, 256] pre-norm

        # Normalize embeddings to match cosine similarity objective
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)  # [B, 256], row-norm 1
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)  # [B, 256], row-norm 1

        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # [B, B]

        ids = batch["id"]  # [B] (or [B, 1] -> squeezed to [B])
        if ids.ndim > 1:
            ids = ids.view(ids.size(0))
        positive_mask = ids.unsqueeze(1) == ids.unsqueeze(0)  # [B, B] bool
        positive_counts = positive_mask.sum(dim=-1, keepdim=True)  # [B, 1]
        targets = positive_mask.float() / positive_counts.clamp_min(1.0)  # [B, B], rows sum to 1

        texts_loss = cross_entropy(logits, targets, reduction='none')  # [B]
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')  # [B]
        loss = (images_loss + texts_loss) / 2.0  # [B]
        return loss.mean()  # scalar []


def cross_entropy(preds, targets, reduction='none'):
    """Full-matrix cross-entropy for soft targets.

    Args:
        preds: [B, B] logits. targets: [B, B] rows sum to 1.
        reduction: 'none' -> [B] per-row; 'mean' -> scalar [].
    Returns:
        [B] or scalar [] loss.
    """
    # preds: [B, B] logits, targets: [B, B] soft targets (rows sum to 1)
    # loss: [B] per-row, or scalar [] if reduction='mean'
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)  # [8, 3, 224, 224]
    input_ids = torch.randint(5, 300, size=(8, 25))  # [8, 25]
    attention_mask = torch.ones(8, 25)  # [8, 25]
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'id': torch.arange(8),
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")
