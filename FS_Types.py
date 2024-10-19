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
BOOLEAN_PARAMS_TERM = ('BOOLEAN', {'default': False, 'tooltip': 'Print params to cmd/terminal.'})
BOOLEAN_PROMPT_TERM = ('BOOLEAN', {'default': False, 'tooltip': 'Print prompt to cmd/terminal.'})

# STRING
STRING_IN = ('STRING', {'default': 'Enter a value.'})
STRING_IN_FORCED = ('STRING', {'forceInput': True})
STRING_ML = ('STRING', {'multiline': True, 'default': 'Enter a value.'})
STRING_WIDGET = ('STRING', {'forceInput': True})

# LLM
LLM_PROMPT_TYPE = (['extra_crispy', 'original', 'both'], {'tooltip': 'extra_crispy = LLM Prompt, original = your typed prompt, both = run prompts'})
MAX_TOKENS_LIST = ('STRING', {'default': '4096', 'tooltip': 'Comma separated list of max_token values to use.'})
Q_LIST = ('STRING', {'default': '1.00', 'tooltip': 'Comma separated list of Q values to use with Clip Attention Multiply.'})
K_LIST = ('STRING', {'default': '1.00', 'tooltip': 'Comma separated list of K values to use with Clip Attention Multiply.'})
V_LIST = ('STRING', {'default': '1.00', 'tooltip': 'Comma separated list of V values to use with Clip Attention Multiply.'})
OUT_LIST = ('STRING', {'default': '1.00', 'tooltip': 'Comma separated list of OUT values to use with Clip Attention Multiply.'})
STRING_PROMPT_POSITIVE = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'Separate prompts with a line\n-----\n...like this.'})
STRING_PROMPT_NEGATIVE = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'Separate prompts with a line\n-----\n...like this.'})
STRING_PROMPT_SELECTED = ('STRING', {
    'multiline': True,
    'dynamicPrompts': True,
    'default':
        'EXAMPLE PROMPT 1 (prompt_type, model, seed, ...)\n'
        '-------------------------\n\n'
        '  POS. PROMPT\n'
        '  ---------------\n'
        '  Selected prompt(s) will show here.\n\n'
        '  NEG. PROMPT\n'
        '  ---------------\n'
        '  Separated like this.\n\n\n'
        'EXAMPLE PROMPT 2 (prompt_type, model, seed, ...)\n'
        '-------------------------\n\n'
        '  POS. PROMPT\n'
        '  ---------------\n'
        '  Selected prompt(s) will show here.\n\n'
        '  NEG. PROMPT\n'
        '  ---------------\n'
        '  Separated like this.'
})

# IMAGE
IMG_WIDTH = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines width input image.'})
IMG_HEIGHT = ('INT', {'default': 1024, 'min': 16, 'max': nodes.MAX_RESOLUTION, 'step': 8, 'tooltip': 'Defines height of the input image.'})
LATENT_BATCH_SIZE = ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': 'The number of latent images in the batch.'})
LATENT_IN = ('LATENT', {'tooltip': 'Input latent image for diffusion process.'})


# SAMPLING
MODEL_IN = ('MODEL', {'tooltip': 'Input Flux or SD model.'})
POSITIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Positive conditioning from clip-encoded text prompt.'})
NEGATIVE_CONDITIONING = ('CONDITIONING', {'tooltip': 'Negative conditioning from clip-encoded text prompt. For SD models only. Will not be used for Flux.'})
SEED = ('INT', {'default': 4, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1, 'tooltip': 'Random noise seed.'})
STEPS = ('INT', {'default': 32, 'min': 1, 'max': 10000, 'tooltip': 'Defines the number of steps to take in the sampling process.'})
ADDED_LINES = ('INT', {'default': 0, 'min': -20, 'max': 50, 'tooltip': 'Add lines to text in image if your prompt is cut off.'})
GUIDANCE = ('FLOAT', {'default': 4.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': 'Controls the influence of external guidance (such as prompts or conditions) on the sampling process.'})
MIN_CFG = ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01, 'tooltip': 'Sets the minimum strength for classifier-free guidance during video generation.'})
MAX_SHIFT = ('FLOAT', {'default': 1.04, 'step':0.01, 'round': 0.01, 'tooltip': 'Defines the maximum pixel movement for image displacement.'})
BASE_SHIFT = ('FLOAT', {'default': 0.44, 'step':0.01, 'round': 0.01, 'tooltip': 'Sets the baseline pixel shift applied before variations.'})
SAMPLING_START_STEP = ('INT', {'default': 0, 'min': 0, 'max': 10000, 'tooltip': 'Step at which to begin sampling.'})
SAMPLING_END_STEP = ('INT', {'default': 32, 'min': 0, 'max': 10000, 'tooltip': 'Step at which to end sampling.'})
DENOISE = ('FLOAT', {
    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
    'tooltip': 'The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.'
})
LATENT_MULT = ('FLOAT', {'default': 1.14, 'min': -10.0, 'max': 10.0, 'step': 0.01, 'tooltip': 'Sets latent multiply factor.'})
FONT_SIZE = ('INT', {'default': 42, 'min': 16, 'max': 96, 'step': 1, 'tooltip': 'Defines burned-in font size.'})

SEED_LIST = ('STRING', {'default': '4', 'tooltip': 'Random noise seed list. If not empty, seed list is used instead of seed.'})
STEPS_LIST = ('STRING', {'default': '32', 'tooltip': 'Defines the number of steps to take in the sampling process. Comma-separated list for multiple runs.'})
GUIDANCE_LIST = ('STRING', {'default': '4.0', 'tooltip': 'Controls the influence of external guidance (such as prompts or conditions) on the sampling process. Comma-separated list for multiple runs.'})
MAX_SHIFT_LIST = ('STRING', {'default': '1.04', 'tooltip': 'Defines the maximum pixel movement for image displacement. Comma-separated list for multiple runs.'})
BASE_SHIFT_LIST = ('STRING', {'default': '0.44', 'tooltip': 'Sets the baseline pixel shift applied before variations. Comma-separated list for multiple runs.'})
DENOISE_LIST = ('STRING', {
    'default': '1.0',
    'tooltip': 'The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling. Comma-separated list for multiple runs.'
})
LATENT_MULT_LIST = ('STRING', {'default': '1.14', 'tooltip': 'Sets latent multiply factor. Comma-separated list for multiple runs.'})

# FVD
FVD_VID_FRAMES = ('INT', {'default': 24, 'min': 1, 'max': 4096, 'tooltip': 'Defines number of frames in the output video.'})
FVD_MOTION_BUCKET = ('INT', {'default': 124, 'min': 1, 'max': 1023, 'tooltip': 'Defines a level of motion present in the output video.'})
FVD_FPS = ('INT', {'default': 12, 'min': 1, 'max': 1024, 'tooltip': 'Defines frames per second in the output video.'})
FVD_AUG_LVL = ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10.0, 'step': 0.01, 'tooltip': 'Adjusts the intensity of augmentations applied to the image frames during video conditioning.'})
FVD_EXTEND_CT = ('INT', {'default': 1, 'min': 1, 'max': 100, 'step': 1, 'tooltip': 'Extend video by running sampler again on last video frame. Max 100; uses last frame of each new batch.'})

# DIRECTORIES
try:
    UNET_DIR = (folder_paths.get_filename_list('diffusion_models'), {'tooltip': 'UNET model list.'})
except:
    UNET_DIR = (folder_paths.get_filename_list('unet'), {'tooltip': 'UNET model list.'})

NF4_DIR = (folder_paths.get_filename_list('checkpoints'), {'tooltip': 'Checkpoint model list.'})
CKPT_DIR = (folder_paths.get_filename_list('checkpoints'), {'tooltip': 'Checkpoint model list.'})
CLIP_DIR = (folder_paths.get_filename_list('clip'), {'tooltip': 'CLIP model list.'})


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
MODEL_UNIFIED = ('MODEL', 'CLIP', 'VAE', 'INT', 'STRING' )
CONDITIONING = ('CONDITIONING', )

SAMPLER_UNIFIED = ('IMAGE', 'LATENT', )
SAMPLER_FVD = ('LATENT', 'IMAGE', )

STRING_OUT = ('STRING', )
STRING_OUT_2 = ('STRING', 'STRING', )
FS_LLM_OUT = ('CONDITIONING', 'CONDITIONING', 'STRING', 'STRING', )

LATENT = ('LATENT', )
LATENT_CHOOSER = ('LATENT', 'IMAGE', 'INT', 'INT', )
IMAGE = ('IMAGE', )

