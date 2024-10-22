# Project: FlowState Unified Model Loader
# Description: Load checkpoints and UNETs, includes NF4 support.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Load Unified Model Loader node.')


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

import os, sys, time, io
import folder_paths
import warnings
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd

from comfy.utils import load_torch_file
from nodes import UNETLoader
from nodes import CheckpointLoaderSimple

from .NF4Loader import CheckpointLoaderNF4

warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")


##
# NODES
##
class FlowStateUnifiedModelLoader:
    CATEGORY = 'FlowState/loader'
    DESCRIPTION = 'Load checkpoints and UNETs, includes NF4 support.'
    FUNCTION = 'load'
    RETURN_TYPES = MODEL_UNIFIED
    RETURN_NAMES = ('model', 'clip', 'vae', 'seed', 'model_type', )
    OUTPUT_TOOLTIPS = (
        'Checkpoint or UNET model.',
        'The CLIP model used for encoding text prompts.',
        'The VAE model used for encoding and decoding images to and from latent space.',
        'Global seed.',
        'Type of model to use.',
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'nf4_model': NF4_DIR,
                'sd_model': CKPT_DIR,
                'unet_model': UNET_DIR,
                'weight_dtype': (['default', 'fp8_e4m3fn', 'fp8_e5m2'], ),
                'model_type': (['NF4', 'UNET', 'SD'],),
                'clip_1': CLIP_DIR,
                'clip_2': CLIP_DIR,
                'clip_type': (['default', 'sdxl', 'sd3', 'flux'], ),
                'vae_name': (['default'] + s.vae_list(), ),
                'seed': SEED,
            }
        }

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list('vae')
        approx_vaes = folder_paths.get_filename_list('vae_approx')
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith('taesd_decoder.'):
                sd1_taesd_dec = True
            elif v.startswith('taesd_encoder.'):
                sd1_taesd_enc = True
            elif v.startswith('taesdxl_decoder.'):
                sdxl_taesd_dec = True
            elif v.startswith('taesdxl_encoder.'):
                sdxl_taesd_enc = True
            elif v.startswith('taesd3_decoder.'):
                sd3_taesd_dec = True
            elif v.startswith('taesd3_encoder.'):
                sd3_taesd_enc = True
            elif v.startswith('taef1_encoder.'):
                f1_taesd_dec = True
            elif v.startswith('taef1_decoder.'):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append('taesd')
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append('taesdxl')
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append('taesd3')
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append('taef1')
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list('vae_approx')

        encoder = next(filter(lambda a: a.startswith('{}_encoder.'.format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def load_vae(self, vae_name):
        vae = None
        vae_path = None
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
                vae_path = self.load_taesd(vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                vae_path = load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=vae_path)
        return vae

    @classmethod
    def load_clip(self, clip_name1, clip_name2, model_type):
        clip_path1 = folder_paths.get_full_path_or_raise("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("clip", clip_name2)
        if model_type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif model_type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif model_type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX

        clip = None
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)

        return clip

    @classmethod
    def select_models(self, nf4_model, sd_model, unet_model, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name):
        model, clip, vae = None, None, None
        model_fname, clip_fname, vae_fname = None, None, None

        if model_type == 'NF4':
            model_fname = nf4_model
            loaded_model = CheckpointLoaderNF4().load_checkpoint(nf4_model)
            model = loaded_model[0]
            if clip_type == 'default':
                clip_fname = 'included'
                clip = loaded_model[1]
            if vae_name == 'default':
                vae_fname = 'included'
                vae = loaded_model[2]

        if model_type == 'SD':
            model_fname = sd_model
            loaded_model = CheckpointLoaderSimple().load_checkpoint(sd_model)
            model = loaded_model[0]
            if clip_type == 'default':
                clip_fname = 'included'
                clip = loaded_model[1]
            if vae_name == 'default':
                vae_fname = 'included'
                vae = loaded_model[2]

        if model_type == 'UNET':
            model_fname = unet_model
            model = UNETLoader().load_unet(unet_model, weight_dtype)[0]
            print(f'\n\n UNET ({len(model)}): {model} \n\n')

        if vae == None and vae_name != 'default':
            vae_fname = vae_name
            vae = self.load_vae(vae_name)

        if clip == None and clip_type != 'default':
            clip_fname = f'{clip_1} & {clip_2}'
            clip = self.load_clip(clip_1, clip_2, clip_type)

        if vae == None:
            vae_fname = self.vae_list()[0]
            vae = self.load_vae(vae_name)

        if clip == None:
            clip_fname = f'{clip_1} & {clip_2}'
            clip = self.load_clip(clip_1, clip_2, 'flux')

        return model, clip, vae, model_fname, clip_fname, vae_fname

    def load(self, nf4_model, sd_model, unet_model, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name, seed):
        print(
            f'\n\nFlowState Unified Model Loader'
            f'\n  - Preparing loader\n'
        )

        start_time = time.time()

        model, clip, vae, model_name, clip_name, vae_name = self.select_models(nf4_model, sd_model, unet_model, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name)

        loading_duration = time.time() - start_time
        loading_mins = int(loading_duration // 60)
        loading_secs = int(loading_duration - loading_mins * 60)

        print(
            f'\nFlowState Unified Model Loader - Loading complete.'
            f'\n  - Model Name: {model_name}'
            f'\n  - VAE Name: {vae_name}'
            f'\n  - CLIP Name: {clip_name}'
            f'\n  - Loading Time: {loading_mins}m {loading_secs}s\n'
        )

        model_type_out = 'SD' if model_type == 'SD' else 'FLUX'

        return (model, clip, vae, seed, [model_type_out], )


