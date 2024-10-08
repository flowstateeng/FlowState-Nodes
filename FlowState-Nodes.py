# Project: ComfyUI Load & Encode Node
# Description: Combines functionality of the Load Image & VAE Encode Nodes
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


import torch
import numpy as np

import os
import hashlib
import node_helpers
import folder_paths

from PIL import Image, ImageOps, ImageSequence


class FlowStateLoadAndEncode:
    CATEGORY = "image"
    FUNCTION = "load_and_encode"
    RETURN_TYPES = ("LATENT", )

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()

        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "vae": ("VAE", )
            },
        }

    def load_and_encode(self, image, vae):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        encoded = vae.encode(output_image[:,:,:,:3])
        return ({"samples": encoded}, )

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True





NODE_CLASS_MAPPINGS = {
    'FlowStateLoadAndEncode': FlowStateLoadAndEncode,
    'FlowStateEncode': FlowStateEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'FlowStateLoadAndEncode': 'FS Load & Encode',
    'FlowStateEncode': 'FS VAE Encode'
}
