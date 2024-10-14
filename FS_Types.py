# Project: FlowState Types
# Description: Global types for all nodes.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import sys
import nodes
import folder_paths
import copy

##
# ANY TYPE
##
class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

ANY = (AnyType('*'), {})


##
# INPUT TYPES
##
# NUMERICAL
FLOAT = ('FLOAT', {'default': 1, 'min': -sys.float_info.max, 'max': sys.float_info.max, 'step': 0.01})
FLOAT_CLIP_ATTN = ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01})
INT = ('INT', {'default': 1, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1})
MAX_TOKENS = ('INT', {'default': 4096, 'min': 1, 'max': 8192})

# LOGICAL
BOOLEAN = ('BOOLEAN', {'default': True})
BOOLEAN_FALSE = ('BOOLEAN', {'default': False})
BOOLEAN_TRUE = ('BOOLEAN', {'default': True})
BOOLEAN_PARAMS = ('BOOLEAN', {'default': False, 'tooltip': 'Add params to output images.'})
BOOLEAN_PROMPT = ('BOOLEAN', {'default': False, 'tooltip': 'Add prompt to output images.'})

# STRING
STRING_IN = ('STRING', {'default': 'Enter a value.'})
STRING_IN_FORCED = ('STRING', {'forceInput': True})
STRING_ML = ('STRING', {'multiline': True, 'default': 'Enter a value.'})
STRING_PROMPT = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'A stunning image.'})
STRING_PROMPT_LLM = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'Generated LLM prompt will show here.'})
STRING_WIDGET = ('STRING', {'forceInput': True})

# IMAGE
IMG_WIDTH = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines width input image.'})
IMG_HEIGHT = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines height of the input image.'})
LATENT_BATCH_SIZE = ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': 'The number of latent images in the batch.'})
LATENT_IN = ('LATENT', {'tooltip': 'Input latent image for diffusion process.'})
LATENT_MULT = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

# SAMPLING
MODEL_IN = ('MODEL', {'tooltip': 'Input Flux or SD model.'})
POSITIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Positive conditioning from clip-encoded text prompt.'})
NEGATIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Negative conditioning from clip-encoded text prompt. For SD models only. Will not be used for Flux.'})
SEED = ('INT', {'default': 4, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1, 'tooltip': 'Random noise seed.'})
STEPS = ('INT', {'default': 32, 'min': 1, 'max': 10000, 'tooltip': 'Defines the number of steps to take in the sampling process.'})
GUIDANCE = ('FLOAT', {'default': 4.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': 'Controls the influence of external guidance (such as prompts or conditions) on the sampling process.'})
MIN_CFG = ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': 'Sets the minimum strength for classifier-free guidance during video generation.'})
MAX_SHIFT = ('FLOAT', {'default': 1.04, 'step':0.1, 'round': 0.01, 'tooltip': 'Defines the maximum pixel movement for image displacement.'})
BASE_SHIFT = ('FLOAT', {'default': 0.44, 'step':0.1, 'round': 0.01, 'tooltip': 'Sets the baseline pixel shift applied before variations.'})
SAMPLING_START_STEP = ('INT', {'default': 0, 'min': 0, 'max': 10000, 'tooltip': 'Step at which to begin sampling.'})
SAMPLING_END_STEP = ('INT', {'default': 32, 'min': 0, 'max': 10000, 'tooltip': 'Step at which to end sampling.'})
DENOISE = ('FLOAT', {
    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
    'tooltip': 'The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.'
})

# FVD
FVD_VID_FRAMES = ('INT', {'default': 24, 'min': 1, 'max': 4096, 'tooltip': 'Defines number of frames in the output video.'})
FVD_MOTION_BUCKET = ('INT', {'default': 124, 'min': 1, 'max': 1023, 'tooltip': 'Defines a level of motion present in the output video.'})
FVD_FPS = ('INT', {'default': 12, 'min': 1, 'max': 1024, 'tooltip': 'Defines frames per second in the output video.'})
FVD_AUG_LVL = ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10.0, 'step': 0.01, 'tooltip': 'Adjusts the intensity of augmentations applied to the image frames during video conditioning.'})
FVD_EXTEND_CT = ('INT', {'default': 1, 'min': 1, 'max': 100, 'step': 1, 'tooltip': 'Extend video by running sampler again on last video frame. Max 100; uses last frame of each new batch.'})

# DIRECTORIES
try:
    UNET_DIR = (copy.deepcopy(folder_paths.get_filename_list('diffusion_models')), {'tooltip': 'Diffusion model list.'}, )
except:
    UNET_DIR = (copy.deepcopy(folder_paths.get_filename_list('unet')), {'tooltip': 'Diffusion model list.'}, )

NF4_DIR = (copy.deepcopy(folder_paths.get_filename_list('checkpoints')), {'tooltip': 'Checkpoint model list.'}, )
CKPT_DIR = (copy.deepcopy(folder_paths.get_filename_list('checkpoints')), {'tooltip': 'Checkpoint model list.'}, )
CLIP_DIR = (copy.deepcopy(folder_paths.get_filename_list('clip')), {'tooltip': 'CLIP model list.'}, )

# MODEL
CLIP_IN = ('CLIP', {'tooltip': 'The CLIP model used for encoding the text.'})
VAE_IN = ('VAE', {'tooltip': 'The VAE model used for encoding and decoding images.'})


# MISC
JSON_WIDGET = ('JSON', {'forceInput': True})
METADATA_RAW = ('METADATA_RAW', {'forceInput': True})


##
# OUTPUT TYPES
##
MODEL = ('MODEL', )
MODEL_UNIFIED = ('MODEL', 'CLIP', 'VAE', )
CONDITIONING = ('CONDITIONING', )

SAMPLER_UNIFIED = ('LATENT', 'IMAGE', )
SAMPLER_FVD = ('LATENT', 'IMAGE', )

STRING_OUT = ('STRING', )
STRING_OUT_2 = ('STRING', 'STRING', )
FS_LLM_OUT = ('CONDITIONING', 'CONDITIONING', 'STRING', 'STRING', )

LATENT = ('LATENT', )
LATENT_CHOOSER = ('LATENT', 'IMAGE', 'INT', 'INT', )
IMAGE = ('IMAGE', )

