import os, folder_paths

##
# CONSTANTS
##
MAX_RESOLUTION=16384
GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, 'llm_gguf')
DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'
FS_LLM_PROMPT_TAG = '@FLOWSTATE_LLM_PROMPT'
