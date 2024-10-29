# Project: FlowState Latent Chooser
# Description: Select from input/imported images to create a new batch of latent images, or select an empty latent.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'    - Loaded Latent Chooser node.')


##
# FS IMPORTS
##
from .FS_Assets import *
from .FS_Constants import *
from .FS_Types import *
from .FS_Utils import *


##
# OUTSIDE IMPORTS
##
import torch
import numpy as np

import os, sys
import node_helpers
import folder_paths

from PIL import Image, ImageOps, ImageSequence
from comfy import model_management


##
# NODES
##
class FlowStateLatentChooser:
    CATEGORY = 'FlowState/latent'
    DESCRIPTION = 'Create a new batch of latent images to be denoised via sampling.'
    FUNCTION = 'create_latent'
    RETURN_TYPES = LATENT_CHOOSER
    RETURN_NAMES = ('latent', 'vae', 'image', 'width', 'height', )
    OUTPUT_TOOLTIPS = (
        'The latent image batch.',
        'VAE to pass to Styler or Sampler.',
        'The image batch.',
        'Image width.',
        'Image height.',
    )

    @classmethod
    def __init__(self):
        self.device = model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            'required': {
                'model_type': FS_MODEL_TYPE_LIST,
                'custom_width': IMG_WIDTH,
                'custom_height': IMG_HEIGHT,
                'resolution': LATENT_CHOOSER_RESOLUTION,
                'orientation': LATENT_CHOOSER_ORIENTATION,
                'latent_type': (['empty_latent', 'input_img', 'imported_img'],),
                'image': (sorted(files), {'image_upload': True}),
                'vae': VAE_IN
            },
            'optional': {
                'pixels': IMAGE
            }
        }

    @classmethod
    def generate(self, width, height, selected_model, batch_size=1):
        latent_channels = 16 if selected_model == 'FLUX' else 4
        latent = torch.zeros([batch_size, latent_channels, height // 8, width // 8], device=self.device)
        return latent

    @classmethod
    def VALIDATE_INPUTS(s, image, vae=None):
        if not folder_paths.exists_annotated_filepath(image):
            return 'Invalid image file: {}'.format(image)
        return True

    @classmethod
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

            image = i.convert('RGB')

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
                mask = torch.zeros((64,64), dtype=torch.float32, device='cpu')
            output_images.append(image)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        encoded = vae.encode(output_image[:,:,:,:3])
        return encoded, output_image

    def create_latent(self, model_type, resolution, orientation, latent_type, image, vae, custom_width, custom_height, pixels=None):
        print(f'\n\n\nFlowState Latent Chooser - {model_type}')

        selected_model = model_type if isinstance(model_type, str) else model_type[0]
        loaded_latent, loaded_img = self.load_and_encode(image, vae)

        using_custom = resolution == 'custom'
        horizontal_img = orientation == 'horizontal'
        using_empty = latent_type == 'empty_latent'
        using_input = latent_type == 'input_img'
        have_pixels = pixels != None

        width_to_use = loaded_img.shape[2]
        height_to_use = loaded_img.shape[1]
        res = None
        res_split = None

        if using_custom and using_empty:
            width_to_use = custom_width
            height_to_use = custom_height

        if not using_custom and using_empty:
            res_split = res.split(' - ')[0].split('x')
            width_to_use = int(res_split[0]) if horizontal_img else int(res_split[1])
            height_to_use = int(res_split[1]) if horizontal_img else int(res_split[0])

        if using_input and have_pixels:
            width_to_use = pixels.shape[2]
            height_to_use = pixels.shape[1]

        if latent_type == 'empty_latent':
            print(f'  - Preparing empty latent.\n')
            generated_latent = self.generate(width_to_use, height_to_use, selected_model)
            return ({'samples': generated_latent}, vae, loaded_img, width_to_use, height_to_use, )
        elif latent_type == 'input_img':
            print(f'  - Preparing latent from input image.')
            input_latent = vae.encode(pixels[:,:,:,:3])
            return ({'samples': input_latent}, vae, pixels, width_to_use, height_to_use, )
        else:
            print(f'  - Preparing latent from imported image.')
            return ({'samples': loaded_latent}, vae, loaded_img, width_to_use, height_to_use, )

