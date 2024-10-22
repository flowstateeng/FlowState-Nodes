# Project: FlowState Unified Sampler
# Description: One sampler to rule them all.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Loaded Unified Sampler.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FS_Constants import *
from .FS_Assets import *


##
# OUTSIDE IMPORTS
##
import time, copy, itertools, math

import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import warnings

import comfy.utils
import comfy.sd

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_model_advanced import ModelSamplingFlux
from comfy_extras.nodes_latent import LatentMultiply
from comfy_extras.nodes_latent import LatentBatch
from node_helpers import conditioning_set_values

from nodes import common_ksampler


warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning)


##
# NODES
##
class FlowStateUnifiedSampler:
    CATEGORY = 'FlowState/sampler'
    DESCRIPTION = ('Applies Flux or SD model to input conditioning to produce an image.')
    FUNCTION = 'execute'
    RETURN_TYPES = SAMPLER_UNIFIED
    RETURN_NAMES = ('images', 'latents', 'fs_params' )
    OUTPUT_TOOLTIPS = (
        'The image batch.',
        'The latent batch.',
        'The parameters used for the image batch.',
    )

    def __init__(self):
        self.prev_params = []
        self.last_latent_batch = None
        self.last_img_batch = None

    @classmethod
    def INPUT_TYPES(s):
        sampler_list = comfy.samplers.KSampler.SAMPLERS
        selected_samplers = [
            'euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_fast', 'dpm_adaptive',
            'ddpm', 'deis', 'ddim', 'uni_pc'
        ]
        valid_selections = [sampler for sampler in selected_samplers if sampler in sampler_list]
        expanded_sampler_selections = []
        for r in range(1, len(valid_selections) + 1):
            combinations = itertools.combinations(valid_selections, r)
            expanded_sampler_selections.extend([", ".join(comb) for comb in combinations])
        expanded_sampler_list = sampler_list + expanded_sampler_selections

        scheduler_list = comfy.samplers.KSampler.SCHEDULERS
        expanded_scheduler_list = []
        for sc in range(1, len(scheduler_list) + 1):
            combinations = itertools.combinations(scheduler_list, sc)
            expanded_scheduler_list.extend([', '.join(comb) for comb in combinations])

        return {
            'required': {
                'model_type': FS_MODEL_TYPE_LIST,
                'model': MODEL_IN,
                'latent_batch': LATENT_IN,
                'vae': VAE_IN,
                'positive_conditioning': POSITIVE_CONDITIONING,
                'positive_prompt': STRING_PROMPT_POSITIVE,
                'seed': SEED,
                'sampling_algorithm': (expanded_sampler_list, {'tool_tip': 'The sampling algorithm(s) used during the diffusion process.'}, ),
                'scheduling_algorithm': (expanded_scheduler_list, {'tool_tip': 'The scheduling algorithm(s) used during the diffusion process.'}, ),
                'guidance': GUIDANCE_LIST,
                'steps': STEPS_LIST,
                'denoise': DENOISE_LIST,
                'max_shift': MAX_SHIFT_LIST,
                'base_shift': BASE_SHIFT_LIST,
                'multiplier': LATENT_MULT_LIST,
                'add_params': BOOLEAN_PARAMS,
                'add_prompt': BOOLEAN_PROMPT,
                'show_params_in_terminal': BOOLEAN_PARAMS_TERM,
                'show_prompt_in_terminal': BOOLEAN_PROMPT_TERM,
                'font_size': FONT_SIZE,
            },
            'optional': {
                'negative_conditioning': NEGATIVE_CONDITIONING,
                'negative_prompt': STRING_PROMPT_NEGATIVE,
                'added_lines': ADDED_LINES,
                'seed_str_list': SEED_LIST,
            }
        }

    def format_num(self, num, num_type, alt):
        num_to_use = num
        try:
            num_to_use = num_type(num_to_use)
        except:
            num_to_use = alt

        return num_to_use

    def reset(self):
        self.prev_params = []
        self.last_latent_batch = None
        self.last_img_batch = None

    def sample_flux(self, seed, model, positive_conditioning, guidance, sampling_algorithm, scheduling_algorithm,
                    steps, denoise, latent_batch, max_shift, base_shift, width, height):

        randnoise = Noise_RandomNoise(seed)
        patched_model = ModelSamplingFlux().patch(model, max_shift, base_shift, width, height)[0]
        conditioning = conditioning_set_values(positive_conditioning, {'guidance': guidance})
        guider = BasicGuider().get_guider(patched_model, conditioning)[0]
        sampler = comfy.samplers.sampler_object(sampling_algorithm)
        sigmas = BasicScheduler().get_sigmas(patched_model, scheduling_algorithm, steps, denoise)[0]
        return SamplerCustomAdvanced().sample(randnoise, guider, sampler, sigmas, latent_batch)[1]['samples']

    def sample_sd(self, model, seed, steps, guidance, sampling_algorithm, scheduling_algorithm, positive_conditioning, negative_conditioning, latent_batch, denoise):
        return common_ksampler(
            model, seed, steps, guidance, sampling_algorithm, scheduling_algorithm, positive_conditioning, negative_conditioning, latent_batch, denoise=denoise
        )[0]['samples']

    def split_prompt(self, prompt, max_char_count):
        have_negative_prompt = prompt['negative'] != None and prompt['negative'] != 'NEGATIVE PROMPT' and prompt['negative'] != ''

        prompt_str = 'positive: ' + prompt['positive']

        if have_negative_prompt:
            prompt_str += f'; negative: {prompt["negative"]}'

        prompt_str = f'prompt: [{prompt_str}]'

        words = prompt_str.split()
        current_line = []
        lines = []
        current_length = 0

        for word in words:
            if current_length + len(word) + len(current_line) <= max_char_count:
                current_line.append(word)
                current_length += len(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def split_params(self, params, max_line_length):
        params_copy = copy.deepcopy(params)
        prompt_only = len(params_copy) == 1

        prompt_lines = []
        if 'prompt' in params_copy:
            prompt_lines = self.split_prompt(params_copy['prompt'], max_line_length)
            del params_copy['prompt']

        if prompt_only:
            return prompt_lines
        else:
            params_copy['sampling_duration'] = round(params_copy['sampling_duration'], 2)
            params_string = ', '.join([f'{key}: {value}' for key, value in params_copy.items()])

            if len(prompt_lines) > 0:
                params_string += (', ' + ', '.join(prompt_lines))

            lines = []
            current_line = ''

            for param in params_string.split(', '):
                if len(current_line) + len(param) + (1 if current_line else 0) > max_line_length:
                    lines.append(current_line)
                    current_line = param
                else:
                    current_line += (', ' + param) if current_line else param

            if current_line:
                lines.append(current_line)

            return lines

    def add_params(self, img_batch, params, width, height, font_size=42, added_lines=0):
        params_copy = copy.deepcopy(params)

        if 'add_params' in params_copy:
            del params_copy['add_params']

        if 'add_prompt' in params_copy:
            del params_copy['add_prompt']

        using_prompt = 'prompt' in params_copy
        using_params = len(params_copy) > 1
        using_params_but_not_prompt = using_params and not using_prompt
        using_prompt_but_not_params = using_prompt and not using_params
        num_lines = 7 if using_params_but_not_prompt else (7 if using_prompt_but_not_params else 14)

        print(
            f'\nFlowState Unified Sampler - Adding Params.'
            f'\n  - Adding Prompt: {using_prompt}'
        )

        start_time = time.time()

        # Loop over the batch of images
        updated_img_batch_list = []
        for img in img_batch:
            image_np = img.numpy()
            img_min = image_np.min()
            img_max = image_np.max()

            image_norm = (image_np - img_min) / (img_max - img_min) * 255
            image_int = image_norm.astype(np.uint8)
            image = Image.fromarray(image_int)

            # Add text
            font = ImageFont.truetype(FONT_PATH, font_size)
            bbox = font.getbbox('A')
            char_width = bbox[2] - bbox[0]
            line_height = font.getmetrics()[1]
            max_line_len = width // char_width - 2

            # Split parameters into lines of text
            wrapped_text = self.split_params(params_copy, max_line_len - 1)

            # Create a new image with space for text at the bottom
            params_bar_height = (math.ceil(num_lines / 4) * 4 + added_lines) * font_size
            updated_img = Image.new('RGB', (width, height + params_bar_height), (0, 0, 0))
            updated_img.paste(image, (0, 0))

            # Draw text on the image
            draw = ImageDraw.Draw(updated_img)
            y_text = height + font_size // 2

            for line in wrapped_text:
                draw.text((char_width, y_text), line, font=font, fill=(255, 255, 255))
                y_text += line_height + font_size

            # Append updated image to the batch list
            updated_img_batch_list.append(updated_img)

        # Convert the list of PIL images back to a 4D tensor and permute to (n_imgs, height + bar_height, width, 3)
        updated_img_batch_tensor = torch.stack([F.to_tensor(img).permute(1, 2, 0) for img in updated_img_batch_list])

        params_duration = time.time() - start_time
        params_mins = int(params_duration // 60)
        params_secs = int(params_duration - params_mins * 60)

        print(f'  - Complete. Params Duration: {params_mins}m {params_secs}s\n')

        # Return the updated 4D tensor
        return updated_img_batch_tensor

    def check_params(self, params, params_num):
        num_prev_params = len(self.prev_params)
        have_prev_params = num_prev_params > 0
        no_prev_params = not have_prev_params

        adding_params = params['add_params'] == True
        adding_prompt = params['add_prompt'] == True

        in_range = -num_prev_params <= params_num < num_prev_params
        more_imgs = not in_range

        first_batch = no_prev_params or more_imgs or self.last_latent_batch == None or self.last_img_batch == None

        actions = []

        if first_batch:
            print(f'  - First Run.')
            actions.append('run')
            if adding_params: actions.append('add_params')
            if adding_prompt: actions.append('add_prompt')
            self.reset()
            return actions, params

        new_params_stashed_copy = copy.deepcopy(params)
        new_params_working_copy = copy.deepcopy(params)

        prev_params_stashed_copy = copy.deepcopy(self.prev_params[params_num])
        prev_params_working_copy = copy.deepcopy(self.prev_params[params_num])

        for k, v in prev_params_stashed_copy.items():
            if k.startswith('llm_'):
                del prev_params_working_copy[k]

        for k, v in prev_params_stashed_copy.items():
            if k.startswith('llm_') and k in new_params_working_copy:
                del new_params_working_copy[k]

        del new_params_working_copy['add_params']
        del new_params_working_copy['add_prompt']

        del prev_params_working_copy['add_params']
        del prev_params_working_copy['add_prompt']
        del prev_params_working_copy['sampling_duration']

        prev_params_added = prev_params_stashed_copy['add_params'] == True
        prev_prompt_added = prev_params_stashed_copy['add_prompt'] == True
        prev_params_not_added = not prev_params_added
        prev_prompt_not_added = not prev_prompt_added

        new_params_added = params['add_params'] == True
        new_prompt_added = params['add_prompt'] == True
        new_params_not_added = not new_params_added
        new_prompt_not_added = not new_prompt_added

        running = prev_params_working_copy != new_params_working_copy
        not_running = not running

        if running:
            actions.append('run')

        if not_running:
            new_params_working_copy['sampling_duration'] = prev_params_stashed_copy['sampling_duration']
            new_params_stashed_copy['sampling_duration'] = prev_params_stashed_copy['sampling_duration']

        if new_params_added and prev_params_not_added:
            actions.append('add_params')

        if new_params_not_added and prev_params_added:
            actions.append('remove_params')

        if new_params_added and prev_params_added:
            actions.append('keep_params')

        if new_prompt_added and prev_prompt_not_added:
            actions.append('add_prompt')

        if new_prompt_not_added and prev_prompt_added:
            actions.append('remove_prompt')

        if new_prompt_added and prev_prompt_added:
            actions.append('keep_prompt')


        no_actions_taken = len(actions) == 0
        if no_actions_taken:
            return None, new_params_working_copy

        return actions, new_params_stashed_copy

    def sample(self, run_num, num_runs, model_type, model, positive_conditioning, negative_conditioning, positive_prompt, negative_prompt,
               latent_img, vae, seed, guidance, sampling_algorithm, scheduling_algorithm, steps, denoise, max_shift, base_shift, multiplier,
               add_params, add_prompt, font_size, fs_llm_params, show_params_in_terminal, show_prompt_in_terminal, added_lines):

        print(
            f'\n\nFlowState Unified Sampler'
            f'\n  - Preparing run: ({run_num}/{num_runs})'
        )

        start_time = time.time()

        width = latent_img['samples'].shape[3] * 8
        height = latent_img['samples'].shape[2] * 8

        params = {
            'model_type': model_type,
            'seed': seed,
            'width': width,
            'height': height,
            'sampler': sampling_algorithm,
            'scheduler': scheduling_algorithm,
            'steps': steps,
            'guidance': guidance,
            'max_shift': max_shift,
            'base_shift': base_shift,
            'denoise': denoise,
            'multiplier': multiplier,
            'add_params': add_params,
            'add_prompt': add_prompt,
            'prompt': {
                'positive': positive_prompt,
                'negative': negative_prompt if model_type == 'SD' else None
            }
        }
        fvd_out_params = copy.deepcopy(params)
        del fvd_out_params['model_type']
        del fvd_out_params['denoise']
        del fvd_out_params['steps']
        del fvd_out_params['add_params']
        del fvd_out_params['add_prompt']
        del fvd_out_params['prompt']
        fvd_out_params['model'] = model
        fvd_out_params['vae'] = vae
        fvd_out_params['positive_conditioning'] = positive_conditioning

        if fs_llm_params:
            fs_llm_params_split = fs_llm_params.split(',')
            for fs_llm_param in fs_llm_params_split:
                param_split = fs_llm_param.split(':')
                k, v = param_split[0], param_split[1]
                params[k] = v

        if model_type == 'SD':
            del params['max_shift']
            del params['base_shift']

        log_params = copy.deepcopy(params)
        del log_params['model_type']
        del log_params['add_params']
        del log_params['add_prompt']
        del log_params['prompt']

        run_num = -num_runs + (run_num - 1)
        actions, print_params = self.check_params(params, run_num)
        print(f'  - Taking Actions: {actions}')
        print(f'  - Running Model: {model_type}')

        if show_params_in_terminal:
            print(f'  - Params:\n    - {log_params}\n')

        if show_prompt_in_terminal:
            print(f'  - Prompt:\n    - {params["prompt"]}\n')

        if actions == None:
            return (self.last_latent_batch, self.last_img_batch, )

        running = 'run' in actions

        latent_out = None
        img_out = None

        if running:
            if model_type == 'FLUX':
                latent_out = self.sample_flux(
                    seed, model, positive_conditioning, guidance, sampling_algorithm, scheduling_algorithm,
                    steps, denoise, latent_img, max_shift, base_shift, width, height
                )
            else:
                latent_out = self.sample_sd(
                    model, seed, steps, guidance, sampling_algorithm, scheduling_algorithm, positive_conditioning,
                    negative_conditioning, latent_img, denoise
                )

            print(
                f'\nFlowState Unified Sampler - Sampling Complete.'
                f'\n  - Decoding Batch: {latent_out.shape}\n'
            )

            latent_out = LatentMultiply().op({'samples': latent_out}, multiplier)[0]['samples']
            img_out = vae.decode(latent_out)

            params['sampling_duration'] = print_params['sampling_duration'] = time.time() - start_time
            sampling_mins = int(params['sampling_duration'] // 60)
            sampling_secs = int(params['sampling_duration'] - sampling_mins * 60)

            print(
                f'\nFlowState Unified Sampler - Decoding complete.'
                f'\n  - Total Generated Images: {img_out.shape[0]}'
                f'\n  - Output Resolution: {img_out.shape[2]} x {img_out.shape[1]}'
                f'\n  - Generation Time: {sampling_mins}m {sampling_secs}s\n'
            )

        if not running:
            params['sampling_duration'] = print_params['sampling_duration']
            latent_out = self.last_latent_batch[run_num].unsqueeze(0)
            img_out = vae.decode(latent_out)


        adding_params = 'add_params' in actions
        adding_prompt = 'add_prompt' in actions
        removing_params = 'remove_params' in actions
        removing_prompt = 'remove_prompt' in actions
        keeping_params = 'keep_params' in actions
        keeping_prompt = 'keep_prompt' in actions

        add_params_only = (adding_params or keeping_params) and (not keeping_prompt and not adding_prompt)
        add_prompt_only = (adding_prompt or keeping_prompt) and (not keeping_params and not adding_params)
        add_both = (adding_params or keeping_params) and (adding_prompt or keeping_prompt)
        add_neither = (not keeping_params and not adding_params) and (not keeping_prompt and not adding_prompt)

        if add_params_only:
            print_params_copy = copy.deepcopy(print_params)
            del print_params_copy['prompt']
            img_out = self.add_params(img_out, print_params_copy, width, height, font_size, added_lines)

        if add_prompt_only:
            img_out = self.add_params(img_out, {'prompt': print_params['prompt']}, width, height, font_size, added_lines)

        if add_both:
            img_out = self.add_params(img_out, print_params, width, height, font_size, added_lines)

        return img_out, latent_out, params, fvd_out_params

    def execute(self, model_type, model, positive_conditioning, negative_conditioning, positive_prompt, negative_prompt, latent_batch,
                vae, seed, added_lines, seed_str_list, sampling_algorithm, scheduling_algorithm, guidance, steps, denoise, max_shift, base_shift,
                multiplier, add_params, add_prompt, show_params_in_terminal, show_prompt_in_terminal, font_size):

        print(
            f'\n\n\nFlowState Unified Sampler'
            f'\n  - Preparing sampler'
        )

        img_out = []
        latent_out = []
        params_out = []
        fvd_params_out = []

        pos_cond_list = positive_conditioning if isinstance(positive_conditioning, list) else [positive_conditioning]
        pos_prompt_list = positive_prompt if isinstance(positive_prompt, list) else [positive_prompt]

        neg_cond_list = negative_conditioning if isinstance(negative_conditioning, list) else [negative_conditioning]
        neg_prompt_list = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt]

        selected_model = model_type if isinstance(model_type, str) else model_type[0]

        if selected_model != 'SD':
            neg_cond_list = [None]
            neg_prompt_list = [None]

        seed_list = seed_str_list.replace(' ', '').split(',')
        guidance_list = guidance.replace(' ', '').split(',')
        sampler_list = sampling_algorithm.replace(' ', '').split(',')
        scheduler_list = scheduling_algorithm.replace(' ', '').split(',')
        step_list = steps.replace(' ', '').split(',')
        denoise_list = denoise.replace(' ', '').split(',')
        max_shift_list = max_shift.replace(' ', '').split(',')
        base_shift_list = base_shift.replace(' ', '').split(',')
        multiplier_list = multiplier.replace(' ', '').split(',')

        num_runs = len(seed_list) * len(guidance_list) * len(sampler_list) * len(scheduler_list) * len(step_list) \
                * len(denoise_list) * len(max_shift_list) * len(base_shift_list) * len(multiplier_list) * len(pos_cond_list)

        run_num = 1
        for prompt_num, pos_cond in enumerate(pos_cond_list):
            pos_prompt = pos_prompt_list[prompt_num]
            neg_prompt = None
            neg_cond = None
            if selected_model == 'SD': neg_cond = neg_cond_list[prompt_num]
            if selected_model == 'SD': neg_prompt = neg_prompt_list[prompt_num]

            is_fs_llm_prompt = pos_prompt.startswith(f'{FS_LLM_PROMPT_TAG}-')
            fs_llm_params = None
            if is_fs_llm_prompt:
                pos_prompt = pos_prompt.replace(f'{FS_LLM_PROMPT_TAG}-', '')
                prompt_parts = pos_prompt.split('-----')
                fs_llm_params = prompt_parts[0]
                pos_prompt = prompt_parts[1]

            for seed_num, run_seed in enumerate(seed_list):
                seed_to_use = self.format_num(run_seed, int, seed)

                for guidance_num, run_guidance in enumerate(guidance_list):
                    guidance_to_use = self.format_num(run_guidance, float, 4.0)

                    for sampler_num, run_sampler in enumerate(sampler_list):

                        for scheduler_num, run_scheduler in enumerate(scheduler_list):

                            for step_num, run_step in enumerate(step_list):
                                step_to_use = self.format_num(run_step, int, 10)

                                for denoise_num, run_denoise in enumerate(denoise_list):
                                    denoise_to_use = self.format_num(run_denoise, float, 1.0)

                                    for max_shift_num, run_max_shift in enumerate(max_shift_list):
                                        max_shift_to_use = self.format_num(run_max_shift, float, 1.04)

                                        for base_shift_num, run_base_shift in enumerate(base_shift_list):
                                            base_shift_to_use = self.format_num(run_base_shift, float, 1.0)

                                            for multiplier_num, run_multiplier in enumerate(multiplier_list):
                                                multiplier_to_use = self.format_num(run_multiplier, float, 1.0)

                                                batch_img, batch_latent, batch_params, fvd_params = self.sample(
                                                    run_num, num_runs, selected_model, model, pos_cond, neg_cond, pos_prompt, neg_prompt, latent_batch, vae,
                                                    seed_to_use, guidance_to_use, run_sampler, run_scheduler, step_to_use, denoise_to_use, max_shift_to_use,
                                                    base_shift_to_use, multiplier_to_use, add_params, add_prompt, font_size, fs_llm_params, show_params_in_terminal,
                                                    show_prompt_in_terminal, added_lines
                                                )

                                                img_out.append(batch_img)
                                                latent_out.append(batch_latent)
                                                params_out.append(batch_params)
                                                fvd_params_out.append(fvd_params)

                                                # Run number adjust
                                                run_num += 1

        self.last_latent_batch = torch.cat(latent_out)
        self.last_img_batch = torch.cat(img_out)
        self.prev_params += params_out

        return (self.last_img_batch, {'samples': self.last_latent_batch}, fvd_params_out, )


