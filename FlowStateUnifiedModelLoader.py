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
from .FS_Assets import *
from .FS_Constants import *
from .FS_Types import *
from .FS_Utils import *


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
                'model_file': ALL_MODEL_LISTS(),
                'weight_dtype': (['default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], ),
                'model_type': (['NF4', 'UNET', 'SD'],),
                'clip_1': CLIP_LIST(),
                'clip_2': CLIP_LIST(),
                'clip_type': (['default', 'sdxl', 'sd3', 'flux'], ),
                'vae_name': VAE_LIST(),
                'seed': SEED,
            }
        }

    def load_taesd(self, name):
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

    def select_models(self, model_file, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name):
        model, clip, vae = None, None, None
        clip_fname, vae_fname = None, None

        is_nf4 = model_type == 'NF4'
        is_sd = model_type == 'SD'
        is_unet = model_type == 'UNET'

        default_clip = clip_type == 'default'
        default_vae = vae_name == 'default'

        model_loader = CheckpointLoaderNF4 if is_nf4 else (CheckpointLoaderSimple if is_sd else UNETLoader)

        loaded_model = None
        if is_unet:
            loaded_model = model = model_loader().load_unet(model_file, weight_dtype)[0]
        else:
            loaded_model = model_loader().load_checkpoint(model_file)
            model = loaded_model[0]

        clip_and_vae_included = (isinstance(loaded_model, list) or isinstance(loaded_model, tuple)) and len(loaded_model) > 2

        if not default_clip and not default_vae:
            clip_fname = f'{clip_1} & {clip_2}'
            clip_weight_type = 'flux' if default_clip else clip_type
            clip = self.load_clip(clip_1, clip_2, clip_weight_type)
            vae_fname = vae_name
            vae = self.load_vae(vae_fname)
        else:
            if clip_and_vae_included:
                if default_clip:
                    clip_fname = 'included'
                    clip = loaded_model[1]
                if default_vae:
                    vae_fname = 'included'
                    vae = loaded_model[2]
            else:
                clip_fname = f'{clip_1} & {clip_2}'
                clip_weight_type = 'flux' if default_clip else clip_type
                clip = self.load_clip(clip_1, clip_2, clip_weight_type)
                vae_fname = VAE_LIST_PATH[0]
                vae = self.load_vae(vae_fname)

        return model, clip, vae, clip_fname, vae_fname

    def load(self, model_file, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name, seed):
        print(
            f'\n\nFlowState Unified Model Loader'
            f'\n  - Preparing loader\n'
        )

        start_time = time.time()

        model, clip, vae, clip_fname, vae_fname = self.select_models(
            model_file, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name
        )

        loading_duration, loading_mins, loading_secs = get_mins_and_secs(start_time)
        vae_warn = '(Selected VAE not available)' if vae_fname != vae_name else ''

        print(
            f'\nFlowState Unified Model Loader - Loading complete.'
            f'\n  - Model Name: {model_file}'
            f'\n  - VAE Name: {vae_fname} {vae_warn}'
            f'\n  - CLIP Name: {clip_fname}'
            f'\n  - Loading Time: {loading_mins}m {loading_secs}s\n'
        )

        model_type_out = 'SD' if model_type == 'SD' else 'FLUX'

        return (model, clip, vae, seed, [model_type_out], )


