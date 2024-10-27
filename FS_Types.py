# Project: FlowState Types
# Description: Global types for all nodes.
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng



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
import sys
import nodes


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
LLM_PROMPT_COMBINATIONS = (['one_to_one', 'combinations'], {
    'tooltip': (
        'Prompt Combination Type\n\n'
        ' - one_to_one: if number of pos prompts == number of neg prompts, run each pos prompt with a single neg prompt\n\n'
        ' - combinations: run all possible combinations of pos and neg prompt.\n\n'
    )
})

LLM_PROMPT_TYPE = (['extra_crispy', 'original', 'both'], {
    'tooltip': (
        f'LLM Prompt Type\n\n'
        f' - extra_crispy = Load local LLM GGUF model to generate prompt.\n\n'
        f' - original = Your originally typed prompt.\n\n'
        f' - both = Your originally typed prompt & the local LLM GGUF model to generate prompt.\n\n'
        # f' - ollama = Connect to your local/remote Ollama service. Requires an "instruct" model.\n\n'
    )
})

clip_preset_vals = '\n\n'.join([f' - {preset}: {vals}' for preset, vals in CLIP_PRESETS.items()])
CLIP_PRESETS_IN = (['default', 'clarity_boost', 'even_flow', 'subtle_focus', 'sharp_detail'], {
    'tooltip': f'CLIP Attention Preset Values (q, k, v, out):\n\n{clip_preset_vals}'
}, )

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
LATENT_CHOOSER_RESOLUTION = ([
    'custom',
    # HORIZONTAL
    '1920x1080 - 16:9',
    '1280x720 - 16:9',
    '1280x768 - 5:3',
    '1280x960 - 4:3',
    '1024x768 - 4:3',
    '2048x512 - 4:1',
    '1152x896 - 9:7',
    '4096x2048 - 2:1',
    '2048x1024 - 2:1',
    '1564x670 - 21:9',
    '2212x948 - 21:9',
    # SQUARE
    '4096x4096 - 1:1',
    '2048x2048 - 1:1',
    '1024x1024 - 1:1',
    '720x720 - 1:1',
    '512x512 - 1:1',
    # VERTICAL
    '1080x1920 - 9:16',
    '720x1280 - 9:16',
    '768x1280 - 3:5',
    '960x1280 - 3:4',
    '768x1024 - 3:4',
    '512x2048 - 1:4',
    '896x1152 - 7:9',
    '2048x4096 - 1:2',
    '1024x2048 - 1:2',
    '670x1564 - 9:21',
    '948x2212 - 9:21',
    ], {
    'tooltip': (
        f'LLM Prompt Type\n\n'
        f' - custom = Specify the exact resolution you want.\n\n'
        f' - preset resolution = Will use this resolution instead of the input resolution.\n\n'
    )
})


# SAMPLING
FS_MODEL_TYPE_LIST = (['FLUX', 'SD'], )
FS_MODEL_TYPE_STR = ('STRING', {
    'default': 'FLUX', 'tooltip': 'Model type passed from Unified Model Loader or other string. If not empty, model_type is used instead of the dropdown model type field.'
})
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
    'tooltip': (
        f'Sampler Denoise Amount\n\n'
        f' - The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\n\n'
    )
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
    'tooltip': (
        f'Sampler Denoise Amount\n\n'
        f' - The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\n\n'
        f' - Comma-separated list for multiple runs.\n\n'
    )
})
LATENT_MULT_LIST = ('STRING', {'default': '1.14', 'tooltip': 'Sets latent multiply factor. Comma-separated list for multiple runs.'})

# FVD
FLUX_MODEL_IN = ('MODEL', {'tooltip': 'Input Flux model.'})
FS_PARAMS_IN = ('FS_PARAMS', {'tooltip': 'Flux sampler parameters.'})
FS_SELECTED_IMG = ('INT', {'default': 1, 'min': 1, 'max': 10, 'tooltip': 'Image that was selected from the Flux batch.'})
FVD_VID_FRAMES = ('INT', {'default': 24, 'min': 1, 'max': 4096, 'tooltip': 'Defines number of frames in the output video.'})
FVD_MOTION_BUCKET = ('INT', {'default': 124, 'min': 1, 'max': 1023, 'tooltip': 'Defines a level of motion present in the output video.'})
FVD_FPS = ('INT', {'default': 12, 'min': 1, 'max': 1024, 'tooltip': 'Defines frames per second in the output video.'})

FVD_AUG_LVL = ('FLOAT', {
    'default': 0.0, 'min': 0.0, 'max': 10.0, 'step': 0.01,
    'tooltip': 'Adjusts the intensity of augmentations applied to the image frames during video conditioning.'
})

FVD_EXTEND_CT = ('INT', {
    'default': 1, 'min': 1, 'max': 100, 'step': 1,
    'tooltip': 'Extend video by running sampler again on last video frame. Max 100; uses last frame of each new batch.'
})


# MODEL
CLIP_IN = ('CLIP', {'tooltip': 'The CLIP model used for encoding the text.'})
VAE_IN = ('VAE', {'tooltip': 'The VAE model used for encoding and decoding images.'})
CONTROL_NET_IN = ('CONTROL_NET', {'tooltip': 'The Control Net model used to patch your image model.'})

UNET_LIST = (UNET_LIST_PATH, {'tooltip': 'UNET model list.'})
NF4_LIST = (NF4_LIST_PATH, {'tooltip': 'Checkpoint model list.'})
CKPT_LIST = (CKPT_LIST_PATH, {'tooltip': 'Checkpoint model list.'})
CLIP_LIST = (CLIP_LIST_PATH, {'tooltip': 'CLIP model list.'})
VAE_LIST = (VAE_LIST_PATH, {'tooltip': 'VAE model list.'})
CONTROL_NET_LIST = (['none'] + CONTROL_NET_LIST_PATH, {'tooltip': 'Control Net model list.'})
LORA_LIST = (['none'] + LORA_LIST_PATH, {'tooltip': 'LoRA model list.'})
ALL_MODEL_LISTS = (ALL_MODEL_LIST_PATHS, {'tooltip': 'All models list.'})


# STYLE MODELS
# style_list = ['none', 'control_net', 'lora', 'control_net & none', 'lora & none', 'control_net & lora', 'all three']
style_list = ['none', 'lora', 'lora & none']
STYLE_TYPE = (style_list, {
    'tooltip': (
        f'Unified Styler Type\n\n'
        f' - none: runs your original model as is.\n\n'
        f' - control net: applies selected control net with specified settings.\n\n'
        f' - lora: applies selected control net with specified settings.\n\n'
        f' - control net & lora: 2 runs - one with control net & one with lora.\n\n'
        f' - control net & none: 2 runs - one with control net & one with no style models applied.\n\n'
        f' - lora & none: 2 runs - one with lora & one with no style models applied.\n\n'
        f' - all three: 3 runs - one with control net, one with lora & one with no style models applied.\n\n'
    )
})
CONTROL_NET_STRNGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'Control Net Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 10.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for strengths.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
CONTROL_NET_START = ('STRING', {'default': '0.0',
    'tooltip': (
        f'Control Net Start\n\n'
        f' - default: 0.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 1.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for starts.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
CONTROL_NET_END = ('STRING', {'default': '1.0',
    'tooltip': (
        f'Control Net End\n\n'
        f' - default: 1.0\n\n'
        f' - min: 0.0\n\n'
        f' - max: 1.0\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for ends.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
CANNY_THRESHOLD_LOW = ('STRING', {'default': '0.4',
    'tooltip': (
        f'Canny Threshold Low\n\n'
        f' - default: 0.4\n\n'
        f' - min: 0.01\n\n'
        f' - max: 0.99\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for low thresholds.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
CANNY_THRESHOLD_HIGH = ('STRING', {'default': '0.8',
    'tooltip': (
        f'Canny Threshold High\n\n'
        f' - default: 0.8\n\n'
        f' - min: 0.01\n\n'
        f' - max: 0.99\n\n'
        f' * if using multiple control nets, use the following comma-separated list format for high thresholds.\n\n'
        f' - 1 control net: 0.4\n\n'
        f' - 2 control nets: 0.4, 0.5\n\n'
        f' - 3 control nets: 0.4, 0.5, 0.6\n\n'
    )
})
LORA_STRENGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'LoRA Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: -100.0\n\n'
        f' - max: 100.0\n\n'
        f' * if using multiple loras, use the following comma-separated list format for strengths.\n\n'
        f' - 1 lora: 0.4\n\n'
        f' - 2 loras: 0.4, 0.5\n\n'
        f' - 3 loras: 0.4, 0.5, 0.6\n\n'
    )
})
LORA_CLIP_STRENGTH = ('STRING', {'default': '1.0',
    'tooltip': (
        f'LoRA CLIP Strength\n\n'
        f' - default: 1.0\n\n'
        f' - min: -100.0\n\n'
        f' - max: 100.0\n\n'
        f' * if using multiple loras, use the following comma-separated list format for CLIP strengths.\n\n'
        f' - 1 lora: 0.4\n\n'
        f' - 2 loras: 0.4, 0.5\n\n'
        f' - 3 loras: 0.4, 0.5, 0.6\n\n'
    )
})


# MISC
JSON_WIDGET = ('JSON', {'forceInput': True})
METADATA_RAW = ('METADATA_RAW', {'forceInput': True})


##
# OUTPUT TYPES
##
MODEL = ('MODEL', )
MODEL_UNIFIED = ('MODEL', 'CLIP', 'VAE', 'INT', ['FLUX', 'SD'], )
CONDITIONING = ('CONDITIONING', )

SAMPLER_UNIFIED = ('IMAGE', 'LATENT', 'FS_PARAMS', )
SAMPLER_FVD = ('LATENT', 'IMAGE', )

STRING_OUT = ('STRING', )
STRING_OUT_2 = ('STRING', 'STRING', )
FS_PROMPT_OUT = ('CONDITIONING', 'CONDITIONING', 'STRING', 'STRING', 'CLIP', )

LATENT = ('LATENT', )
LATENT_CHOOSER = ('LATENT', 'VAE', 'IMAGE', 'INT', 'INT', )
IMAGE = ('IMAGE', )

STYLER_UNIFIED = ('MODEL', 'VAE', 'CONDITIONING', 'CONDITIONING', 'STRING', 'STRING', 'IMAGE', 'IMAGE', )

