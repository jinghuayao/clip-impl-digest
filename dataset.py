import os
import cv2
import torch
import albumentations as A

import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    """Image+caption dataset returning tokenized pairs with image ids.

    Args:
        image_filenames: array-like len N (repeats allowed for multi-captions).
        captions: array-like len N raw strings.
        ids: array-like len N int image ids (same id = same image).
        tokenizer: HF tokenizer; encodes full corpus once to [N, S].
        transforms: albumentations pipeline -> [224, 224, 3].

    __getitem__(idx) returns input_ids [S], attention_mask [S],
        image [3, 224, 224] float32, caption str, id scalar [].
    """
    def __init__(self, image_filenames, captions, ids, tokenizer, transforms):
        """See class docstring; tokenizes all captions up front."""

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.ids = ids
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        """Load and transform one pair.

        Args:
            idx: row index 0 <= idx < N.
        Returns:
            dict: input_ids [S], attention_mask [S], image [3, 224, 224],
                caption str, id scalar [].
        """
        # encoded_captions each: [N, S] (N = dataset len, S = padded len of full corpus)
        # returns: input_ids [S], attention_mask [S], image [3, 224, 224],
        #          caption str, id scalar []
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")  # [H, W, 3] BGR uint8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # [H, W, 3] RGB
        image = self.transforms(image=image)['image']  # [224, 224, 3] float32 normalized
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # [3, 224, 224]
        item['caption'] = self.captions[idx]
        item['id'] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item


    def __len__(self):
        """Return N = number of caption rows."""
        return len(self.captions)



def get_transforms(mode="train"):
    """Build resize+normalize pipeline.

    Args:
        mode: 'train' or 'valid' (currently identical).
    Returns:
        albumentations Compose: HWC uint8 -> [224, 224, 3] float32.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    
