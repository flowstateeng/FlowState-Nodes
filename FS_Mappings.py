from .FS_Nodes import *


##
# SYSTEM STATUS
##
print(f'  - Loading node name mappings...')


##
# MAPPINGS
##
NODE_CLASS_MAPPINGS = {
    # 'FlowStateFVDSampler': FlowStateFVDSampler,
    # 'FlowStateUnifiedSampler': FlowStateUnifiedSampler,
    'FlowStateUnifiedModelLoader': FlowStateUnifiedModelLoader,
    'FlowStatePromptLLM': FlowStatePromptLLM,
    'FlowStatePromptLLMOutput': FlowStatePromptLLMOutput,
    'FlowStateLatentChooser': FlowStateLatentChooser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 'FlowStateFVDSampler': 'FlowState FVD Sampler',
    # 'FlowStateUnifiedSampler': 'FlowState Unified Sampler',
    'FlowStateUnifiedModelLoader': 'FlowState Unified Model Loader',
    'FlowStatePromptLLM': 'FlowState LLM Prompt',
    'FlowStatePromptLLMOutput': 'FlowState LLM Prompt Output',
    'FlowStateLatentChooser': 'FlowState Latent Chooser'
}


##
# SYSTEM STATUS
##
print(f'  - Mappings Loaded. Available nodes:')

for fs_node in NODE_CLASS_MAPPINGS:
    print(f'    - {fs_node}: {NODE_CLASS_MAPPINGS[fs_node]}')

