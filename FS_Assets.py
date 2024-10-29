# Project: FlowState Assets
# Description: Paths to assets needed by nodes.
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
from .FS_Utils import *



##
# OUTSIDE IMPORTS
##
import os
import folder_paths


##
# ASSETS
##
WEB_DIRECTORY = './web'
FONT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts/ShareTechMono-Regular.ttf')


UNET_LIST_PATH = None
try:
    UNET_LIST_PATH = folder_paths.get_filename_list('diffusion_models')
    UNET_LIST_PATH = lambda: folder_paths.get_filename_list('diffusion_models')
except ImportError:
    UNET_LIST_PATH = None

if UNET_LIST_PATH == None:
    try:
        UNET_LIST_PATH = folder_paths.get_filename_list('unet')
        UNET_LIST_PATH = lambda: folder_paths.get_filename_list('unet')
    except ImportError:
        UNET_LIST_PATH = None

print(f'    - Setting UNET_LIST_PATH to: {UNET_LIST_PATH}')


NF4_LIST_PATH = lambda: folder_paths.get_filename_list('checkpoints')
CKPT_LIST_PATH = lambda: folder_paths.get_filename_list('checkpoints')
CLIP_LIST_PATH = lambda: folder_paths.get_filename_list('clip')
VAE_LIST_PATH = lambda: get_vae_list()
CONTROL_NET_LIST_PATH = lambda: folder_paths.get_filename_list('controlnet')
LORA_LIST_PATH = lambda: folder_paths.get_filename_list('loras')
ALL_MODEL_LIST_PATHS = lambda: UNET_LIST_PATH() + CKPT_LIST_PATH()

