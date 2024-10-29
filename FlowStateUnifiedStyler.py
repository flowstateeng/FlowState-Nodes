# Project: FlowState Unified Video Sampler
# Description: One video sampler to rule them all.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng



##
# SYSTEM STATUS
##
print(f'    - Loaded Unified Styler node.')


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
import time, copy, gc, warnings
import torch

from nodes import ControlNetLoader
from nodes import ControlNetApplyAdvanced
from nodes import LoraLoader
from comfy_extras.nodes_canny import Canny
import comfy.sd

warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


style_list_types = {
    'none': ['none'],
    'control_net': ['control_net'],
    'lora': ['lora'],
    'control_net & none': ['control_net', 'none'],
    'lora & none': ['lora', 'none'],
    'control_net & lora': ['control_net', 'lora'],
    'all three': ['control_net', 'lora', 'none']
}


##
# NODES
##
class FlowStateUnifiedStyler:
    CATEGORY = 'FlowState/styler'
    DESCRIPTION = 'Loads & applies LoRAs & Control Nets for image generation.'
    FUNCTION = 'apply_style'
    RETURN_TYPES = STYLER_UNIFIED
    RETURN_NAMES = (
        'model',
        'latent_batch',
        'vae',
        'positive_conditioning',
        'negative_conditioning',
        'positive_prompt',
        'negative_prompt',
        'image',
        'cn_image',
    )
    OUTPUT_TOOLTIPS = (
        'The model, with LoRA optionally applied.',
        'Either the latent passed through from the Latent Chooser or the Control Net processed latent.',
        'The VAE to pass to the sampler.',
        'The positive conditioning, with control net optionally applied.',
        'The negative conditioning, with control net optionally applied.',
        'Pass through positive prompt.',
        'Pass through negative prompt.',
        'Input image.',
        'Input image with Canny Preprocessor applied.',
    )

    def __init__(self):
        self.loaded_models = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'style_type': STYLE_TYPE,
                'model': MODEL_IN,
                'latent_batch': LATENT_IN,
                'vae': VAE_IN,
                'image': IMAGE,
                'clip': CLIP_IN,
                'positive_conditioning': POSITIVE_CONDITIONING,
                'negative_conditioning': NEGATIVE_CONDITIONING,
                'positive_prompt': STRING_PROMPT_POSITIVE,
                'negative_prompt': STRING_PROMPT_NEGATIVE,
                # 'controlnet_1': CONTROL_NET_LIST(),
                # 'controlnet_2': CONTROL_NET_LIST(),
                # 'controlnet_3': CONTROL_NET_LIST(),
                # 'controlnet_strength': CONTROL_NET_STRNGTH,
                # 'controlnet_start': CONTROL_NET_START,
                # 'controlnet_end': CONTROL_NET_END,
                # 'canny_low_threshold': CANNY_THRESHOLD_LOW,
                # 'canny_high_threshold': CANNY_THRESHOLD_HIGH,
                'lora_1': LORA_LIST(),
                'lora_2': LORA_LIST(),
                'lora_3': LORA_LIST(),
                'lora_strength': LORA_STRENGTH,
                'clip_strength': LORA_CLIP_STRENGTH,
                # 'unload_loras': (['none'], )
            }
        }

    def populate_lists(lists, params):
        for r, row in enumerate(params):
            for p, param in enumerate(row):
                lists[p] += [param]

    def process_params(self, input_str):
        values = input_str.replace(' ', '').split(',')
        params_str = ';'.join(values)

        for v, val in enumerate(values):
            is_float = '.' in val
            is_int = val.isdigit()

            values[v] = float(val) if is_float or is_int else 1.0

        return values, params_str

    def unload_loras(self):
        gc.collect()
        torch.cuda.empty_cache()

    def apply_controlnet(self, controlnet_name, positive_conditioning, negative_conditioning, image,
        controlnet_strength, controlnet_start, controlnet_end, vae, canny_low_threshold, canny_high_threshold):

        canny_img = Canny().detect_edge(image, canny_low_threshold, canny_high_threshold)[0]
        controlnet = ControlNetLoader().load_controlnet(controlnet_name)[0]
        pos_cond, neg_cond = ControlNetApplyAdvanced().apply_controlnet(
            positive_conditioning, negative_conditioning, controlnet, canny_img, controlnet_strength,
            controlnet_start, controlnet_end, vae=vae, extra_concat=[]
        )
        return pos_cond, neg_cond, canny_img

    # def apply_style(self,
    #         style_type, model, latent_batch, vae, image, positive_conditioning, negative_conditioning,
    #         positive_prompt, negative_prompt, controlnet_1, controlnet_2, controlnet_3, controlnet_strength,
    #         controlnet_start, controlnet_end, canny_low_threshold, canny_high_threshold, lora_1, lora_2, lora_3,
    #         lora_strength, clip, clip_strength, unload_loras
    #     ):

    def apply_style(self,
            style_type, model, latent_batch, vae, image, positive_conditioning, negative_conditioning, positive_prompt, negative_prompt,
            lora_1, lora_2, lora_3, lora_strength, clip, clip_strength
        ):

        print(f'\n\nFlowState Unified Styler - adding selected styles.')

        start_time = time.time()

        lora_input_list = [lora_1, lora_2, lora_3]
        lora_list = [lora_name for lora_name in lora_input_list if lora_name != 'none']
        lora_strength_list, lora_strength_str = self.process_params(lora_strength)
        clip_strength_list, clip_strength_str = self.process_params(clip_strength)

        lora_params_str = ','.join([f'{l}:' + '; '.join(p) for l,p in [ ('lora_list', lora_list) ]]) + ','
        lora_params_str += f'lora_strengths:{lora_strength_str},clip_strengths:{clip_strength_str}'

        lora_strength_list_matches = len(lora_strength_list) == len(lora_list)
        clip_strength_list_matches = len(clip_strength_list) == len(lora_list)

        # cn_input_list = [controlnet_1, controlnet_2, controlnet_3]
        # cn_list = [controlnet_name for controlnet_name in cn_input_list if controlnet_name != 'none']

        # cn_strength_list, cn_strength_str = self.process_params(controlnet_strength)
        # cn_start_list, cn_start_str = self.process_params(controlnet_start)
        # cn_end_list, cn_end_str = self.process_params(controlnet_end)
        # canny_low_list, canny_low_str = self.process_params(canny_low_threshold)
        # canny_high_list, canny_high_str = self.process_params(canny_high_threshold)

        # cn_params_str = ','.join([f'{l}:' + '; '.join(p) for l,p in [ ('cn_list', cn_list) ]]) + ','
        # cn_params_str += (
        #     f'cn_strengths:{cn_strength_str},'
        #     f'cn_starts:{cn_start_str},'
        #     f'cn_ends:{cn_end_str},'
        #     f'cn_lows:{canny_low_str},'
        #     f'cn_highs:{canny_high_str}'
        # )

        # cn_strength_list_matches = len(cn_list) == len(cn_strength_list)
        # cn_start_list_matches = len(cn_list) == len(cn_start_list)
        # cn_end_list_matches = len(cn_list) == len(cn_end_list)
        # canny_low_list_matches = len(cn_list) == len(canny_low_list)
        # canny_high_list_matches = len(cn_list) == len(canny_high_list)

        pos_prompt_list = positive_prompt if isinstance(positive_prompt, list) else [positive_prompt]
        neg_prompt_list = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt]

        pos_cond_list = positive_conditioning if isinstance(positive_conditioning, list) else [positive_conditioning]
        neg_cond_list = negative_conditioning if isinstance(negative_conditioning, list) else [negative_conditioning]

        style_list = style_list_types[style_type]

        styler_out = [[], [], [], [], []]

        for style in style_list:
            for pos_num, pos_cond in enumerate(pos_cond_list):
                neg_cond = neg_cond_list[pos_num]
                base_pos_prompt = pos_prompt_list[pos_num].replace(f'-----', f',style:{style}-----')

                if style == 'none':
                    print(f' - Not applying styles to this image')
                    styler_out[0] += [model]
                    styler_out[1] += [pos_cond]
                    styler_out[2] += [neg_cond]
                    styler_out[4] += [base_pos_prompt]

                # if style == 'control_net':
                #     print(f' - Applying Control Net to this conditioning')
                #     cn_pos_prompt = base_pos_prompt.replace(f'-----', f',{cn_params_str}-----')
                #     cn_out = [pos_cond, neg_cond, None]
                #     for cn_num, controlnet in enumerate(cn_list):
                #         print(f'  - Applying Control {cn_num + 1}: {controlnet}')
                #         first_cn = cn_num == 0
                #         strength = cn_strength_list[cn_num] if cn_strength_list_matches else cn_strength_list[-1]
                #         start = cn_start_list[cn_num] if cn_start_list_matches else cn_start_list[-1]
                #         end = cn_end_list[cn_num] if cn_end_list_matches else cn_end_list[-1]
                #         low = canny_low_list[cn_num] if canny_low_list_matches else canny_low_list[-1]
                #         high = canny_high_list[cn_num] if canny_high_list_matches else canny_high_list[-1]

                #         cn_pos, cn_neg, cn_canny_img = self.apply_controlnet(
                #             controlnet, cn_out[0], cn_out[1], image, strength, start, end, vae, low, high
                #         )
                #         cn_out[0] = cn_pos
                #         cn_out[1] = cn_neg
                #         if first_cn: cn_out[2] = cn_canny_img
                #         else: cn_out[2] += cn_canny_img

                #     styler_out[0] += [model]
                #     styler_out[1] += [cn_out[0]]
                #     styler_out[2] += [cn_out[1]]
                #     styler_out[3] += [cn_out[2]]
                #     styler_out[4] += [cn_pos_prompt]

                if style == 'lora':
                    print(f' - Patching model with LoRA')
                    lora_pos_prompt = base_pos_prompt.replace(f'-----', f',{lora_params_str}-----')
                    patched_model = copy.deepcopy(model)

                    for lora_num, lora_name in enumerate(lora_list):
                        print(f'  - Patching model with LoRA {lora_num + 1}: {lora_name}')

                        lora_strength = lora_strength_list[lora_num] if lora_strength_list_matches else lora_strength_list[-1]
                        clip_strength = clip_strength_list[lora_num] if clip_strength_list_matches else clip_strength_list[-1]

                        patched_model = LoraLoader().load_lora(patched_model, clip, lora_name, lora_strength, clip_strength)[0]

                    styler_out[0] += [patched_model]
                    styler_out[1] += [pos_cond]
                    styler_out[2] += [neg_cond]
                    styler_out[4] += [lora_pos_prompt]

        styler_duration, styler_mins, styler_secs = get_mins_and_secs(start_time)

        # model_mem_sizes = [ m.size for m in styler_out[0] ]
        # kb = [ round(mem / 1024, 2) for mem in model_mem_sizes ]
        # mb = [ round(mem / 1024, 2) for mem in kb ]
        # gb = [ round(mem / 1024, 2) for mem in mb ]

        print(
            f'\n\n\nFlowState Unified Styler - Styling generation complete.'
            f'\n  - Styling Time: {styler_mins}m {styler_secs}s'
            # f'\n  - Model Sizes (bytes): {model_mem_sizes}'
            # f'\n  - Model Sizes (kb): {kb}'
            # f'\n  - Model Sizes (mb): {mb}'
            # f'\n  - Model Sizes (gb): {gb}'
        )

        out_latent = styler_out[3] if len(styler_out[3]) > 0 else latent_batch

        if out_latent != latent_batch:
            for i, img in enumerate(out_latent):
                latent_img = vae.encode(img)[0]
                out_latent[i] = latent_img

        out_canny = torch.cat(styler_out[3]) if len(styler_out[3]) > 0 else image

        return (styler_out[0], out_latent, vae, styler_out[1], styler_out[2], styler_out[4], neg_cond_list, image, out_canny, )

