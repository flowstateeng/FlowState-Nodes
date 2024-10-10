from .FS_Nodes import *

##
# MAPPINGS
##
NODE_CLASS_MAPPINGS = {
    'FlowStateUnifiedModelLoader': FlowStateUnifiedModelLoader,
    'FlowStateLatentChooser': FlowStateLatentChooser,
    'FlowStatePromptLLM': FlowStatePromptLLM,
    'FlowStatePromptLLMOutput': FlowStatePromptLLMOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'FlowStateUnifiedModelLoader': 'FlowState Unified Model Loader',
    'FlowStateLatentChooser': 'FlowState Latent Chooser',
    'FlowStatePromptLLM': 'FlowState LLM Prompt',
    'FlowStatePromptLLMOutput': 'FlowState LLM Prompt Output',
}
