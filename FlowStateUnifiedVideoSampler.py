# Project: FlowState Unified Sampler
# Description: One sampler to rule them all.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'    - Loaded FVD Sampler node.')


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
import time, copy, itertools, math

import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import warnings
import folder_paths

import comfy.utils
import comfy.sd

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_model_advanced import ModelSamplingFlux
from comfy_extras.nodes_latent import LatentMultiply
from comfy_extras.nodes_latent import LatentBatch
from node_helpers import conditioning_set_values

from nodes import common_ksampler

warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# UserWarning: Using padding='same' with even kernel lengths and odd dilation

##
# NODES
##
class FlowStateUnifiedVideoSampler:
    CATEGORY = 'FlowState/sampler'
    DESCRIPTION = 'Loads & applies FVD model to input image to produce a video.'
    FUNCTION = 'sample'
    RETURN_TYPES = SAMPLER_FVD
    RETURN_NAMES = ('latent', 'image', )
    OUTPUT_TOOLTIPS = (
        'The latent image batch.',
        'Image batch.',
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'seed': SEED,
                'fs_params': FS_PARAMS_IN,
                'width': IMG_WIDTH,
                'height': IMG_HEIGHT,
                'ckpt_name': (folder_paths.get_filename_list('checkpoints'), ),
                'sampling_algorithm': (comfy.samplers.KSampler.SAMPLERS, ),
                'scheduling_algorithm': (comfy.samplers.KSampler.SCHEDULERS, ),
                'svd_steps': STEPS,
                'flux_steps': STEPS,
                'svd_min_guidance': MIN_CFG,
                'svd_guidance': GUIDANCE,
                'flux_guidance': GUIDANCE,
                'flux_denoise': DENOISE,
                'init_image': IMAGE,
                'selected_img': FS_SELECTED_IMG,
                'num_video_frames': FVD_VID_FRAMES,
                'motion_type_id': FVD_MOTION_BUCKET,
                'fps': FVD_FPS,
                'augmentation_level': FVD_AUG_LVL,
                'sequence_count': FVD_EXTEND_CT
            }
        }

    @classmethod
    def patch(self, model, min_guidance):
        def linear_cfg(args):
            cond = args['cond']
            uncond = args['uncond']
            cond_scale = args['cond_scale']

            scale = torch.linspace(min_guidance, cond_scale, cond.shape[0], device=cond.device).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        patched_model = model.clone()
        patched_model.set_model_sampler_cfg_function(linear_cfg)
        return patched_model

    @classmethod
    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path_or_raise('checkpoints', ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths('embeddings'))
        return (out[0], out[3], out[2])

    @classmethod
    def encode(self, clip_vision, init_image, vae, width, height, num_video_frames, motion_type_id, fps, augmentation_level):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, 'bilinear', 'center').movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled, {'motion_type_id': motion_type_id, 'fps': fps, 'augmentation_level': augmentation_level, 'concat_latent_image': t}]]
        negative = [[torch.zeros_like(pooled), {'motion_type_id': motion_type_id, 'fps': fps, 'augmentation_level': augmentation_level, 'concat_latent_image': torch.zeros_like(t)}]]
        latent = torch.zeros([num_video_frames, 4, height // 8, width // 8])
        return (positive, negative, {'samples':latent})

    def sample_flux(self, init_img_params, svd_seq, flux_denoise, flux_steps, flux_guidance, num_video_frames):
        seed = init_img_params['seed']
        model = init_img_params['model']
        vae = init_img_params['vae']
        positive_conditioning = init_img_params['positive_conditioning']
        guidance = init_img_params['guidance']
        sampling_algorithm = init_img_params['sampler']
        scheduling_algorithm = init_img_params['scheduler']
        max_shift = init_img_params['max_shift']
        base_shift = init_img_params['base_shift']
        width = init_img_params['width']
        height = init_img_params['height']

        # print(f'\n\n SEQ SVD IMGS: {svd_img.shape}')
        vae_encoded = vae.encode(svd_seq)
        # print(f'\n\n VAE ENCODED: {vae_encoded.shape}')

        randnoise = Noise_RandomNoise(seed)
        patched_model = ModelSamplingFlux().patch(model, max_shift, base_shift, width, height)[0]
        conditioning = conditioning_set_values(positive_conditioning, {'guidance': flux_guidance})
        guider = BasicGuider().get_guider(patched_model, conditioning)[0]
        sampler = comfy.samplers.sampler_object(sampling_algorithm)
        sigmas = BasicScheduler().get_sigmas(patched_model, scheduling_algorithm, flux_steps, flux_denoise)[0]
        flux_sampler = SamplerCustomAdvanced()

        flux_seq_latents, flux_seq_imgs = [], []
        for frame_num in range(num_video_frames):
            print(f'\nFrame: {frame_num + 1} of {num_video_frames}.')
            frame = vae_encoded[frame_num, :, :, :].unsqueeze(0)
            flux_latent = flux_sampler.sample(randnoise, guider, sampler, sigmas, {'samples': frame})[1]['samples']
            flux_img = vae.decode(flux_latent)

            flux_seq_latents.append(flux_latent)
            flux_seq_imgs.append(flux_img)

        flux_seq_latents = torch.cat(flux_seq_latents, dim=0)
        flux_seq_imgs = torch.cat(flux_seq_imgs, dim=0)

        return flux_seq_latents, flux_seq_imgs

    def sample(self, seed, fs_params, width, height, ckpt_name, sampling_algorithm, scheduling_algorithm, svd_steps,
               flux_steps, svd_min_guidance, svd_guidance, flux_guidance, flux_denoise, init_image, selected_img, num_video_frames, motion_type_id,
               fps, augmentation_level, sequence_count):

        print_fs_params = copy.deepcopy(fs_params)[selected_img - 1]
        del print_fs_params['model']
        del print_fs_params['vae']
        del print_fs_params['positive_conditioning']

        print(
            f'\n FlowState FVD Sampler - Loading models.\n'
            f'\n  - seed: {seed}'
            f'\n  - fs_params: {print_fs_params}'
            f'\n  - width: {width}'
            f'\n  - height: {height}'
            f'\n  - sampling_algorithm: {sampling_algorithm}'
            f'\n  - scheduling_algorithm: {scheduling_algorithm}'
            f'\n  - svd_steps: {svd_steps}'
            f'\n  - flux_steps: {flux_steps}'
            f'\n  - min_guidance: {svd_min_guidance}'
            f'\n  - svd_guidance: {svd_guidance}'
            f'\n  - flux_guidance: {flux_guidance}'
            f'\n  - flux_denoise: {flux_denoise}'
            f'\n  - num_video_frames: {num_video_frames}'
            f'\n  - motion_type_id: {motion_type_id}'
            f'\n  - fps: {fps}'
            f'\n  - augmentation_level: {augmentation_level}'
            f'\n  - sequence_count: {sequence_count}\n\n'
        )
        start_time = time.time()

        model_components = self.load_checkpoint(ckpt_name)
        model_patcher = model_components[0]
        clip_vision = model_components[1]
        svd_vae = model_components[2]

        patched_model = self.patch(model_patcher, svd_min_guidance)

        init_img_params = fs_params[selected_img - 1]

        print(f'\n FlowState FVD Sampler - Sampling.')

        sequence_latents = []
        sequence_imgs = []
        for seq_num in range(sequence_count):
            print(f'  - Generating SVD Video Sequence: {seq_num + 1} of {sequence_count}.\n')

            init_frame = init_image if seq_num == 0 else sequence_imgs[-1][num_video_frames - 1, :, :, :].unsqueeze(0)

            encoded_components = self.encode(clip_vision, init_frame, svd_vae, width, height, num_video_frames, motion_type_id, fps, augmentation_level)
            positive_conditioning = encoded_components[0]
            negative_conditioning = encoded_components[1]
            latent_frames = encoded_components[2]

            latent_batch = common_ksampler(
                patched_model, seed, svd_steps, svd_guidance, sampling_algorithm, scheduling_algorithm, positive_conditioning,
                negative_conditioning, latent_frames, denoise=1
            )
            svd_latents = latent_batch[0]['samples']
            svd_imgs = svd_vae.decode(svd_latents)

            print(f'\n FlowState FVD Sampler - Processing frames with Flux.')
            flux_seq_latents, flux_seq_imgs = self.sample_flux(init_img_params, svd_imgs, flux_denoise, flux_steps, flux_guidance, num_video_frames)

            sequence_latents.append(flux_seq_latents)
            sequence_imgs.append(flux_seq_imgs)

        latent_out = {'samples': torch.cat(sequence_latents, dim=0)}

        print(
            f'\n FlowState FVD Sampler - Sampling Complete.'
            f'\n  - Decoding Batch: {latent_out["samples"].shape}\n'
        )

        img_out = torch.cat(sequence_imgs, dim=0)

        sampling_duration = time.time() - start_time
        sampling_mins = int(sampling_duration // 60)
        sampling_secs = int(sampling_duration - sampling_mins * 60)

        print(
            f'\n FlowState FVD Sampler - Generation complete.'
            f'\n  - Total Generated Frames: {latent_out["samples"].shape[0]}'
            f'\n  - Output Resolution: {img_out.shape[1]} x {img_out.shape[2]}\n'
            f'\n  - Generation Time: {sampling_mins}m {sampling_secs}s\n'
        )

        return (latent_out, img_out, )






# no multiple llm gens for same prompt / different settings (qkv)


# x/y output grid
# update node definitions - use anything multi - is_changed - selected_img

