# Project: FlowState Latent Chooser
# Description: Select from input/imported images to create a new batch of latent images, or select an empty latent.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# FS IMPORTS
##
from .FS_Types import *
from .FS_Constants import *
from .FS_Assets import *


##
# OUTSIDE IMPORTS
##
import torch
import numpy as np

import os, sys
import node_helpers
import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

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
    RETURN_NAMES = ('latent', 'width', 'height', )
    OUTPUT_TOOLTIPS = (
        'The latent image batch.',
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
                'width': ('INT', {'default': 512, 'min': 16, 'max': MAX_RESOLUTION, 'step': 8, 'tooltip': 'The width of the latent images in pixels.'}),
                'height': ('INT', {'default': 512, 'min': 16, 'max': MAX_RESOLUTION, 'step': 8, 'tooltip': 'The height of the latent images in pixels.'}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': 'The number of latent images in the batch.'}),
                'latent_type': (['empty_latent', 'input_img', 'imported_img'],),
                'image': (sorted(files), {'image_upload': True}),
                'vae': ('VAE', )
            },
            'optional': {
                'pixels': IMAGE
            }
        }

    @classmethod
    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
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

    def create_latent(self, latent_type, image, vae, width, height, batch_size=1, pixels=None):
        if latent_type == 'empty_latent':
            latent = self.generate(width, height, batch_size)
            return ({'samples':latent}, width, height, )
        elif latent_type == 'input_img':
            latent = vae.encode(pixels[:,:,:,:3])
            img_width = pixels.shape[2]
            img_height = pixels.shape[1]
            return ({'samples':latent}, img_width, img_height, )
        else:
            latent, loaded_img = self.load_and_encode(image, vae)
            img_width = loaded_img.shape[2]
            img_height = loaded_img.shape[1]
            return ({'samples':latent}, img_width, img_height, )


