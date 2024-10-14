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
from .FS_Types import *
from .FS_Constants import *
from .FS_Assets import *


##
# OUTSIDE IMPORTS
##
import time
import torch
import comfy.utils
import comfy.sd
import folder_paths
import warnings

from nodes import common_ksampler

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")


##
# NODES
##
class FlowStateFVDSampler:
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
                # SAMPLER
                'seed': SEED,
                'steps': STEPS,
                'guidance': GUIDANCE,
                'sampling_algo': (comfy.samplers.KSampler.SAMPLERS, ),
                'scheduling_algo': (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": DENOISE,
                # LOADER
                'ckpt_name': (folder_paths.get_filename_list('checkpoints'), ),
                # VIDEO GUIDANCE
                'min_cfg': MIN_CFG,
                # FVD CONDITIONING
                'init_image': IMAGE,
                'width': IMG_WIDTH,
                'height': IMG_HEIGHT,
                'video_frames': FVD_VID_FRAMES,
                'motion_bucket_id': FVD_MOTION_BUCKET,
                'fps': FVD_FPS,
                'augmentation_level': FVD_AUG_LVL,
                'extend_count': FVD_EXTEND_CT
            }
        }

    @classmethod
    def patch(self, model, min_cfg):
        def linear_cfg(args):
            cond = args['cond']
            uncond = args['uncond']
            cond_scale = args['cond_scale']

            scale = torch.linspace(min_cfg, cond_scale, cond.shape[0], device=cond.device).reshape((cond.shape[0], 1, 1, 1))
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
    def encode(self, clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, 'bilinear', 'center').movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled, {'motion_bucket_id': motion_bucket_id, 'fps': fps, 'augmentation_level': augmentation_level, 'concat_latent_image': t}]]
        negative = [[torch.zeros_like(pooled), {'motion_bucket_id': motion_bucket_id, 'fps': fps, 'augmentation_level': augmentation_level, 'concat_latent_image': torch.zeros_like(t)}]]
        latent = torch.zeros([video_frames, 4, height // 8, width // 8])
        return (positive, negative, {'samples':latent})

    def sample(self, seed, steps, guidance, sampling_algo, scheduling_algo, denoise, ckpt_name, min_cfg, init_image,
               width, height, video_frames, motion_bucket_id, fps, augmentation_level, extend_count):

        print(f'\n FlowState FVD Sampler - Loading models.\n')
        start_time = time.time()

        model_components = self.load_checkpoint(ckpt_name)
        model_patcher = model_components[0]
        clip_vision = model_components[1]
        vae = model_components[2]

        patched_model = self.patch(model_patcher, min_cfg)

        batches = []
        for i in range(extend_count):
            print(
                f'\n FlowState FVD Sampler - Sampling.'
                f'\n  - Video Sequence: {i + 1} of {extend_count}.\n'
            )
            init_latent = None if i == 0 else batches[-1][video_frames - 1, :, :, :].unsqueeze(0)
            init_frame = init_image if i == 0 else vae.decode(init_latent)

            encoded_components = self.encode(clip_vision, init_frame, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)
            positive_conditioning = encoded_components[0]
            negative_conditioning = encoded_components[1]
            latent_frames = encoded_components[2]

            latent_batch = common_ksampler(
                patched_model, seed, steps, guidance, sampling_algo, scheduling_algo, positive_conditioning, negative_conditioning, latent_frames, denoise=denoise
            )
            batches.append(latent_batch['samples'])

        latent_out = {'samples': torch.cat(batches, dim=0)}

        print(
            f'\n FlowState FVD Sampler - Sampling Complete.'
            f'\n  - Decoding Batch: {latent_out["samples"].shape}\n'
        )

        img_out = vae.decode(latent_out['samples'])

        print(
            f'\n FlowState FVD Sampler - Generation complete.'
            f'\n  - Total Generated Frames: {latent_out["samples"].shape[0]}'
            f'\n  - Output Resolution: {img_out.shape[1]} x {img_out.shape[2]}\n'
            f'\n  - Generation Time: {time.time() - start_time}\n'
        )

        return (latent_out, img_out, )

