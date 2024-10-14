# Project: FlowState Unified Sampler
# Description: One sampler to rule them all.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Loaded Unified Sampler.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FS_Constants import *
from .FS_Assets import *


##
# OUTSIDE IMPORTS
##
import time

import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import warnings, copy

import comfy.utils
import comfy.sd

from comfy_extras.nodes_custom_sampler import RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_model_advanced import ModelSamplingFlux
from comfy_extras.nodes_latent import LatentMultiply

from nodes import common_ksampler


warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning)


##
# NODES
##
class FlowStateUnifiedSampler:
    CATEGORY = 'FlowState/sampler'
    DESCRIPTION = ('Applies Flux or SD model to input conditioning to produce an image.')
    FUNCTION = 'sample'
    RETURN_TYPES = SAMPLER_UNIFIED
    RETURN_NAMES = ('latent', 'image', )
    OUTPUT_TOOLTIPS = (
        'The latent image batch.',
        'Image batch.',
    )

    def __init__(self):
        self.last_params = None
        self.last_latent_batch = None
        self.last_img_batch = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model_type': (['Flux', 'SD'], ),
                'model': MODEL_IN,
                'latent_batch': LATENT_IN,
                'vae': VAE_IN,
                'positive_conditioning': POSITIVE_CONDITIONING,
                'seed': SEED,
                'guidance': GUIDANCE,
                'sampling_algo': (comfy.samplers.KSampler.SAMPLERS, ),
                'scheduling_algo': (comfy.samplers.KSampler.SCHEDULERS, ),
                'steps': STEPS,
                'denoise': DENOISE,
                'max_shift': MAX_SHIFT,
                'base_shift': BASE_SHIFT,
                'multiplier': LATENT_MULT,
                'add_params': BOOLEAN_PARAMS,
                'add_prompt': BOOLEAN_PROMPT,
            },
            'optional': {
                'negative_conditioning': NEGATIVE_CONDITIONING
            }
        }

    def sample_flux(self, seed, model, positive_conditioning, guidance, sampling_algo, scheduling_algo,
                    steps, denoise, latent_batch, max_shift, base_shift, width, height):

        print(f'  - Sampling Flux Model\n')

        noise = RandomNoise().get_noise(seed)[0]
        shifted_model = ModelSamplingFlux().patch(model, max_shift, base_shift, width, height)[0]
        guided_conditioning = FluxGuidance().append(positive_conditioning, guidance)[0]
        guider = BasicGuider().get_guider(shifted_model, guided_conditioning)[0]
        sampler = KSamplerSelect().get_sampler(sampling_algo)[0]
        sigmas = BasicScheduler().get_sigmas(shifted_model, scheduling_algo, steps, denoise)[0]

        return SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent_batch)

    def sample_sd(self, model, seed, steps, guidance, sampling_algo, scheduling_algo, positive_conditioning, negative_conditioning, latent_batch, denoise):
        print(f'  - Sampling SD Model\n')

        return common_ksampler(
            model, seed, steps, guidance, sampling_algo, scheduling_algo, positive_conditioning, negative_conditioning, latent_batch, denoise=denoise
        )

    def split_params(params, max_line_length):
        params_string = ', '.join([f"{key}: {value}" for key, value in params.items()])
        lines = []
        current_line = ""

        for param in params_string.split(', '):
            if len(current_line) + len(param) + (1 if current_line else 0) > max_line_length:
                lines.append(current_line)
                current_line = param
            else:
                current_line += (', ' + param) if current_line else param

        if current_line:
            lines.append(current_line)
        return lines

    def add_params(self, img_batch, params, width, height, font_size=32):
        using_prompt = True if 'prompt' in params else False

        print(
            f'\nFlowState Unified Sampler - Adding Params.'
            f'\n  - Adding Prompt: {using_prompt} - IMG SHAPE: {img_batch.shape}'
        )

        resize_transform = transforms.Resize([width, height])
        resized_image = resize_transform(img_batch)

        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(resized_image)

        image_np = image_tensor(0).permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255
        image_np = image_np.astype(np.uint8)

        image = Image.fromarray(image_np)
        font = ImageFont.truetype(FONT_PATH, font_size)

        bbox = font.getbbox('A')
        char_width = bbox[2] - bbox[0]
        line_height = font.getmetrics()[1]
        max_line_len = width // char_width

        wrapped_text = self.split_params(params, max_line_len)
        bar_height = len(wrapped_text) * font_size

        updated_img_batch = Image.new('RGB', (width, height + bar_height), (0, 0, 0))
        updated_img_batch.paste(image, (0, 0))

        draw = ImageDraw.Draw(updated_img_batch)

        y_text = height + font_size // 2
        for line in wrapped_text:
            draw.text((10, y_text), line, font=font, fill=(255, 255, 255))
            y_text += line_height + font_size

        return updated_img_batch

    def check_params(self, params):
        no_params = self.last_params['add_params'] == False or self.last_params == None
        no_prompt = self.last_params['add_prompt'] == False or self.last_params == None

        actions = []

        if self.last_params == None or self.last_latent_batch == None or self.last_img_batch == None or self.last_params != params:
            actions.append('run')

        if no_params and params['add_params'] == True:
            actions.append('add_params')

        if self.last_params['add_params'] == True and params['add_params'] == False:
            actions.append('remove_params')

        if no_prompt and params['add_prompt'] == True:
            actions.append('add_prompt')

        if self.last_params['add_prompt'] == True and params['add_prompt'] == False:
            actions.append('remove_prompt')

        return actions


    def sample(self, model_type, model, positive_conditioning, negative_conditioning, latent_batch, vae, seed, guidance,
               sampling_algo, scheduling_algo, steps, denoise, max_shift, base_shift, multiplier, add_params, add_prompt):

        print(
            f'\nFlowState Unified Sampler'
            f'\n  - Preparing sampler'
        )

        start_time = time.time()
        self.last_params = params

        width = latent_batch['samples'].shape[1]
        height = latent_batch['samples'].shape[2]

        prompt = { 'positive': None, 'negative': None }

        if isinstance(positive_conditioning, dict) and 'encoded' in positive_conditioning:
            prompt['positive'] = positive_conditioning['text']

        if negative_conditioning and isinstance(negative_conditioning, dict) and 'encoded' in negative_conditioning:
            prompt['negative'] = negative_conditioning['text']

        params = {
            'seed': seed,
            'width': width,
            'height': height,
            'sampler': sampling_algo,
            'scheduler': scheduling_algo,
            'steps': steps,
            'guidance': guidance,
            'max_shift': max_shift,
            'base_shift': base_shift,
            'denoise': denoise,
            'add_params': add_params,
            'add_prompt': add_prompt
        }

        if add_prompt:
            params['prompt'] = prompt

        actions = self.check_params(params)

        if actions == None:
            return

        if actions == 'run':
            processed_batch = None

            if model_type == 'Flux':
                processed_batch = self.sample_flux(
                    seed, model, positive_conditioning, guidance, sampling_algo, scheduling_algo, steps, denoise,
                    latent_batch, max_shift, base_shift, width, height
                )
            else:
                processed_batch = self.sample_sd(
                    model, seed, steps, guidance, sampling_algo, scheduling_algo, positive_conditioning,
                    negative_conditioning, latent_batch, denoise
                )

            print(
                f'\nFlowState Unified Sampler - Sampling Complete.'
                f'\n  - Decoding Batch: {latent_batch["samples"].shape}'
            )

            latent_mult = LatentMultiply().op(latent_batch, multiplier)
            img_batch = vae.decode(latent_mult[0]['samples'])

            sampling_duration = time.time() - start_time
            sampling_mins = sampling_duration // 60
            sampling_secs = round(sampling_duration - sampling_mins * 60)

            print(
                f'\nFlowState Unified Sampler - Decoding complete.'
                f'\n  - Total Generated Images: {latent_batch["samples"].shape[0]}'
                f'\n  - Output Resolution: {img_batch.shape[2]} x {img_batch.shape[1]}'
                f'\n  - Generation Time: {sampling_mins}m {sampling_secs}s\n'
            )

            params['sampling_duration'] = sampling_duration
            self.last_latent_batch = latent_batch
            self.last_img_batch = img_batch

            print(f'  - ACTIONS: {actions}')

            if 'add_params' in actions and 'add_prompt' in actions:
                img_batch = self.add_params(img_batch, params, width, height)

            if 'add_params' in actions and 'add prompt' not in actions:
                del params['prompt']
                img_batch = self.add_params(img_batch, params, width, height)

            if 'add_prompt' in actions and 'add_params' not in actions:
                img_batch = self.add_params(img_batch, {'prompt': params['prompt']}, width, height)

            return (latent_batch, img_batch, )

