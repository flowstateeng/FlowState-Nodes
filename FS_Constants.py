# Project: FlowState Constants
# Description: Global constants for all nodes.
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import os, folder_paths

##
# CONSTANTS
##
MAX_RESOLUTION=16384
GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, 'llm_gguf')
DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'
FS_LLM_PROMPT_TAG = '@FLOWSTATE_LLM_PROMPT'

CLIP_PRESETS = {
    'default': [1, 1, 1, 1],
    'clarity_boost': [0.9, 0.9, 1.4, 8.4],
    'even_flow': [1, 1, 1, 4],
    'subtle_focus': [0.9, 0.9, 1, 4],
    'sharp_detail': [1, 1, 1, 8.4]
}

