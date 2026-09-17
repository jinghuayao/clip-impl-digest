import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import config as CFG


class ImageEncoder(nn.Module):
    """ResNet50 image backbone (timm, global-avg-pooled).

    Args:
        model_name: timm model name (default CFG.model_name).
        pretrained: load ImageNet weights if True.
        trainable: unfreeze backbone if True, else frozen features.

    Forward:
        x [B, 3, 224, 224] -> [B, 2048].
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """Encode a batch of images.

        Args:
            x: [B, 3, 224, 224] RGB normalized images.
        Returns:
            [B, 2048] pooled features.
        """
        # x: [B, 3, 224, 224] -> out: [B, 2048]
        # (timm resnet50, num_classes=0, global_pool='avg')
        return self.model(x)


class TextEncoder(nn.Module):
    """DistilBERT text backbone; CLS token is the sentence embedding.

    Uses eager attention (MPS fix: SDPA+dropout is unsupported on MPS).
    Args:
        model_name: HF model id (default CFG.text_encoder_model).
        pretrained: load HF weights if True, else random init.
        trainable: unfreeze transformer if True.

    Forward:
        input_ids [B, S], attention_mask [B, S] -> [B, 768].
        S = per-batch padded length (<= CFG.max_length).
    """
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        # NOTE (Mac MPS fix): SDPA + dropout fails on MPS with
        # "scaled_dot_product_attention for MPS does not support dropout",
        # so force eager attention which supports dropout on MPS.
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            config = DistilBertConfig()
            config._attn_implementation = "eager"
            self.model = DistilBertModel(config=config)
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """Encode tokenized captions to CLS embeddings.

        Args:
            input_ids: [B, S] token ids.
            attention_mask: [B, S] (1 = real token, 0 = pad).
        Returns:
            [B, 768] CLS hidden states.
        """
        # input_ids: [B, S], attention_mask: [B, S] (S = padded seq len in batch, <= CFG.max_length)
        # transformer last_hidden_state: [B, S, 768] -> CLS token [:, 0, :]: [B, 768]
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    """Two-layer MLP head mapping encoder feats to shared 256-d space.

    Args:
        embedding_dim: input width (2048 image / 768 text).
        projection_dim: output width (default CFG.projection_dim=256).
        dropout: dropout rate inside the head.

    Forward:
        x [B, embedding_dim] -> [B, projection_dim] (residual + LayerNorm).
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        """Project encoder features to the shared space.

        Args:
            x: [B, embedding_dim].
        Returns:
            [B, projection_dim] normalized-ready embeddings.
        """
        # x: [B, embedding_dim] (2048 for images, 768 for texts)
        # projected: [B, 256] -> fc: [B, 256] -> residual+norm: [B, 256]
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

