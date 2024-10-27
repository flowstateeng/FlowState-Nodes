# Project: FlowState Node Mappings
# Description: Node mappings for ComfyUI registry.
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# FS IMPORTS
##
from .FS_Nodes import *


##
# SYSTEM STATUS
##
print(f'  - Loading node name mappings...')


##
# MAPPINGS
##
NODE_CLASS_MAPPINGS = {
    'FlowStateUnifiedVideoSampler': FlowStateUnifiedVideoSampler,
    'FlowStateUnifiedSampler': FlowStateUnifiedSampler,
    'FlowStateUnifiedModelLoader': FlowStateUnifiedModelLoader,
    'FlowStateUnifiedPrompt': FlowStateUnifiedPrompt,
    'FlowStatePromptOutput': FlowStatePromptOutput,
    'FlowStateLatentChooser': FlowStateLatentChooser,
    'FlowStateUnifiedStyler': FlowStateUnifiedStyler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'FlowStateUnifiedVideoSampler': 'FlowState Unified Video Sampler',
    'FlowStateUnifiedSampler': 'FlowState Unified Sampler',
    'FlowStateUnifiedModelLoader': 'FlowState Unified Model Loader',
    'FlowStateUnifiedPrompt': 'FlowState Unified Prompt',
    'FlowStatePromptOutput': 'FlowState Prompt Output',
    'FlowStateLatentChooser': 'FlowState Latent Chooser',
    'FlowStateUnifiedStyler': 'FlowState Unified Styler'
}


##
# SYSTEM STATUS
##
print(f'  - Mappings Loaded. Available nodes:')

for fs_node in NODE_CLASS_MAPPINGS:
    print(f'    - {fs_node}: {NODE_CLASS_MAPPINGS[fs_node]}')

