from typing import List

import numpy as np
import torch

from PIL import Image

STANDART_MEAN = [0.5, 0.5, 0.5]
STANDART_STD = [0.5, 0.5, 0.5]


def process_images(images: List[Image.Image],
                   rescaling_factor: float,
                   image_shape: tuple,
                   resample=0,
                   mean=STANDART_MEAN, std=STANDART_STD):
    
    images = [img.resize(image_shape, resample) for img in images]
    images = [(np.array(img) * rescaling_factor).astype(np.float32) for img in images]
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    images = [((img - mean) / std).transpose(2, 0, 1) for img in images]
    
    return images
    

def add_image_tokens_to_prompt(prompt, bos_token,
                               n_img_tokens, image_token):
    
    return f"{image_token * n_img_tokens}{bos_token}{prompt}\n"


class PaligemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, n_image_tokens, img_size):
        self.tokenizer = tokenizer 
        self.n_image_tokens = n_image_tokens 
        self.img_size = img_size

        tokens_to_add = {"additional_special_tokens": self.IMAGE_TOKEN}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def __call__(self,
                 text: List[str], images: List[Image.Image],
                 padding: str = "longest", truncation: bool = True) -> dict:

        assert len(images) == 1 and len(text) == 1, \
              f"Received {len(images)} images for {len(text)} prompts."
        
        processed_images = process_images(
            images=images,
            rescaling_factor=1 / 255.0,
            image_shape=(self.img_size, self.img_size),
            resample=Image.Resampling.BICUBIC
        )

        pixel_values = torch.tensor(np.stack(processed_images, axis=0))

        input_strings = [
            add_image_tokens_to_prompt(
                prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                n_img_tokens=self.n_image_tokens,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors='pt',
            padding=padding,
            truncation=truncation
        )
        
        return {"pixel_values": pixel_values, **inputs}