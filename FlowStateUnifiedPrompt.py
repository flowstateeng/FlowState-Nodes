# Project: FlowState LLM Prompt & LLM Output
# Description: Generate an image prompt using an LLM.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Loaded Unified Prompt & Prompt Output nodes.')


##
# FS IMPORTS
##
from .FS_Assets import *
from .FS_Constants import *
from .FS_Types import *
from .FS_Utils import *


##
# LIBRARIES
##
import os, sys, time, copy, itertools
import importlib
import warnings

import comfy.utils
import comfy.sd

Llama = None

try:
    Llama = importlib.import_module('llama_cpp_cuda').Llama
    print(f'    - HAVE LLAMA CUDA')
except ImportError:
    print(f'    - NO LLAMA CUDA')
    Llama = None

if Llama == None:
    try:
        Llama = importlib.import_module('llama_cpp').Llama
        print(f'    - NO LLAMA CUDA, BUT HAVE LLAMA')
    except ImportError:
        print(f'    - NO LLAMA OR CUDA')
        Llama = None


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comfy'))

warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings('ignore', message='1Torch was not compiled with flash attention')
warnings.filterwarnings('ignore', category=FutureWarning)


##
# NODES
##
class FlowStateUnifiedPrompt:
    CATEGORY = 'FlowState/prompt'
    DESCRIPTION = (
        'Generate an image prompt using an LLM. If using multiple parameters, will need to be '
        'paired with a sampler that can accomodate them, like the FlowState Unified Sampler.'
    )
    FUNCTION = 'generate_prompts'
    RETURN_TYPES = FS_PROMPT_OUT
    RETURN_NAMES = (
        'positive_conditioning',
        'negative_conditioning',
        'positive_prompt',
        'negative_prompt',
        'multiplied_clip',
    )
    OUTPUT_TOOLTIPS = (
        'Selected positive conditioning(s) used to guide the diffusion model.',
        'Original negative conditioning(s) used to guide the diffusion model.',
        'Selected positive text prompt(s).',
        'Original negative text prompt(s).',
        'CLIP with CLIP Attention Multiply applied.',
    )

    @classmethod
    def INPUT_TYPES(cls):
        available_models = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            gguf_files = [file for file in os.listdir(GLOBAL_MODELS_DIR) if file.endswith('.gguf')]
            available_models.extend(gguf_files)
        else:
            available_models.append('no available models')

        expanded_model_list = []
        for sc in range(1, len(available_models) + 1):
            combinations = itertools.combinations(available_models, sc)
            expanded_model_list.extend([', '.join(comb) for comb in combinations])

        return {
            'required': {
                'clip': CLIP_IN,
                'positive_text_prompt': STRING_PROMPT_POSITIVE,
                'negative_text_prompt': STRING_PROMPT_NEGATIVE,
                'prompt_combinations': LLM_PROMPT_COMBINATIONS,
                'prompt_type': LLM_PROMPT_TYPE,
                'seed': SEED,
                'clip_preset': CLIP_PRESETS_IN,
                'llm_model': (expanded_model_list, {'tooltip': 'All combinations of your available GGUF models.'}, ),
                'max_tokens': MAX_TOKENS_LIST,
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

    def tag_prompt(self, params, prompt, msg=''):
        condensed_prompt = prompt.replace('\n\n', '. ').replace('\n', '. ').replace('..', '.').replace('. .', '.')
        tagged_prompt = f'{FS_LLM_PROMPT_TAG}-{params}-----{msg + " - "}{condensed_prompt}'
        return tagged_prompt

    def multiply_clip_attn(self, clip, q, k, v, out):
        multiplied = clip.clone()
        state_dict = multiplied.patcher.model_state_dict()

        for key in state_dict:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, q)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, k)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, v)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, out)

        return multiplied

    def encode_text(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([cond, output], )

    def generate_msgs(self, instructions):
        return [
            {'role': 'system', 'content': f'You are a helpful assistant.'},
            {'role': 'user',
            'content': f'An image generation prompt is a single paragraph summary to describe the subject and '
                    f'style of an image. It includes a description of the kind of image, the subject of '
                    f'the image, and some description of the image medium and style in the form of short '
                    f'keyword.\n\nCreate an image generation prompt for the subject \'a creepy creature '
                    f'shadow in the dark in a dimly lit tunnel\' in the style \'digital art illustration '
                    f'with intricate details\'.'
            },
            {'role': 'assistant',
            'content': f'Image Description: A digitally crafted illustration portrays a chilling scene within '
                    f'a dimly lit, cavernous tunnel. The dominant subject of the image is a mysterious '
                    f'creature, its form subtly discernible only as a menacing shadow on the walls of the '
                    f'tunnel. Its elongated silhouette suggests a serpentine creature with sharp '
                    f'protrusions and an ominous aura. The creature\'s eyes, a pair of glowing orbs, appear '
                    f'eerily human-like yet alien. The tunnel is riddled with intricate details that '
                    f'enhance the eerie atmosphere: dust particles floating lazily in the feeble light, '
                    f'ancient and crumbling stone, water droplets sparkling on the damp walls, and a '
                    f'hauntingly beautiful, bioluminescent fungi growing in the shadows. The dimly lit '
                    f'environment is highlighted by strategically placed light sources that create '
                    f'dramatic chiaroscuro effects, casting an unsettling and atmospheric glow on the '
                    f'scene. Digital Art Illustration with Intricate Details (Dark, Atmospheric, '
                    f'Suspenseful)'
            },
            {'role': 'user', 'content': 'Now compile the description and style into a single paragraph summary'},
            {'role': 'assistant',
            'content': f'Digital art illustration featuring a dark, atmospheric, and suspenseful scene within '
                    f'a dimly lit, cavernous tunnel. The subject of the image is a mysterious creature, '
                    f'depicted only as a menacing shadow on the walls, with elongated silhouette and sharp '
                    f'protrusions. The creature\'s eyes, a pair of glowing orbs, appear eerily human-like '
                    f'yet alien. The tunnel is adorned with intricate details, such as dust particles, '
                    f'ancient crumbling stone, water droplets, and a hauntingly beautiful bioluminescent '
                    f'fungi growing in the shadows. Dramatic chiaroscuro effects are created through '
                    f'strategically placed light sources, casting an unsettling and atmospheric glow on '
                    f'the scene.'
            },
            {'role': 'user', 'content': f'create a detailed summary without the title or style'},
            {'role': 'assistant',
            'content': f'A dimly lit, cavernous tunnel is the setting for this chilling digital illustration. '
                    f'A mysterious creature lurks in the shadows, its elongated silhouette suggestive of a '
                    f'serpentine creature with sharp protrusions and an ominous aura. The creature\'s eyes, '
                    f'a pair of glowing orbs, appear eerily human-like yet alien. The tunnel is riddled '
                    f'with intricate details that enhance the eerie atmosphere: dust particles floating '
                    f'lazily in the feeble light, ancient and crumbling stone, water droplets sparkling on '
                    f'the damp walls, and a hauntingly beautiful, bioluminescent fungi growing in the '
                    f'shadows. Dramatic chiaroscuro effects are created through strategically placed light '
                    f'sources, casting an unsettling and atmospheric glow on the scene.'
            },
            {'role': 'user',
            'content': f'Generate a prompt from \'magical garden, sitting on a blue_bench, Disney Princess in '
                    f'pink_dress, blonde_hair, portrait, Pixar, Disney style, photorealistic, high '
                    f'details, detailed skin, pores, freckles\''
            },
            {'role': 'assistant',
            'content': f'In a captivating, Pixar-inspired, Disney-style, photorealistic portrait, a magical '
                    f'garden unfolds as a breathtaking backdrop. The subject of the image is a Disney '
                    f'Princess, dressed elegantly in a pink gown with flowing fabric, sitting serenely on '
                    f'a blue bench. The princess boasts long, blonde hair and a radiant smile. The garden '
                    f'is meticulously detailed, with vibrant, enchanted flora and fauna, as well as '
                    f'whimsical details like sparkling fairy lights and a picturesque waterfall. The '
                    f'princess is highlighted against the lush, detailed foliage, with a focus on the '
                    f'high-definition details of her porcelain skin, visible freckles, and the intricacies '
                    f'of her elegant gown. The image is rendered in the captivating, photorealistic style '
                    f'that exemplifies both the Disney and Pixar brands, capturing the princess\'s timeless '
                    f'beauty and the magic of her enchanting surroundings.'
            },
            {'role': 'user', 'content': instructions},
        ]

    def load_gguf_llm(self, pos_prompt, multiplied_clip, max_tokens, seed, llm_model_path):
        model_path = os.path.join(GLOBAL_MODELS_DIR, llm_model_path)
        loaded_model = Llama(model_path=model_path, n_gpu_layers=-1, seed=seed, verbose=False, n_ctx=2048,)

        generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 50, 'repeat_penalty': 1.2}
        llm_messages = self.generate_msgs(f'Generate a prompt from {pos_prompt}')

        llm_prompt = loaded_model.create_chat_completion(llm_messages, **generate_kwargs)
        llm_prompt_cleaned = llm_prompt['choices'][0]['message']['content'].strip()

        llm_conditioning = self.encode_text(multiplied_clip, llm_prompt_cleaned)

        return llm_prompt_cleaned, llm_conditioning

    def generate_prompt(self, multiplied_clip, positive_txt, negative_txt, prompt_type, seed, llm_model,
                        max_tokens, q, k, v, out, prompts_out_copy):

        param_string = (
            f'llm_prompt_type:{prompt_type},'
            f'llm_seed:{seed},'
            f'llm_model:{llm_model},'
            f'llm_max_tokens:{max_tokens},'
            f'q:{q},'
            f'k:{k},'
            f'v:{v},'
            f'out:{out}'
        )

        if prompt_type == 'original':
            param_string = param_string.replace(f'llm_seed:{seed},llm_model:{llm_model},llm_max_tokens:{max_tokens},', '')

        neg_prompt = negative_txt.replace('\n\n', '. ').replace('\n', '. ')
        pos_prompt = positive_txt.replace('\n\n', '. ').replace('\n', '. ')

        accum_pos_cond_list = prompts_out_copy[0]
        accum_neg_cond_list = prompts_out_copy[1]
        accum_pos_txt_list = prompts_out_copy[2]
        accum_neg_txt_list = prompts_out_copy[3]
        accum_orig_pos_list = prompts_out_copy[4]
        accum_orig_neg_list = prompts_out_copy[5]

        repeated_pos_txt = positive_txt in accum_orig_pos_list
        repeated_neg_txt = negative_txt in accum_orig_neg_list

        have_llama = Llama != None
        selected_llm = prompt_type == 'extra_crispy' or prompt_type == 'both'
        is_valid_llm = have_llama and llm_model.endswith('.gguf')

        repeated_pos_txt_idx = accum_orig_pos_list.index(positive_txt) if repeated_pos_txt else None
        repeated_neg_txt_idx = accum_orig_neg_list.index(negative_txt) if repeated_neg_txt else None

        pos_txt = accum_pos_txt_list[repeated_pos_txt_idx] if repeated_pos_txt else pos_prompt
        neg_txt = accum_neg_txt_list[repeated_neg_txt_idx] if repeated_neg_txt else neg_prompt
        pos_cond = accum_pos_cond_list[repeated_pos_txt_idx] if repeated_pos_txt else self.encode_text(multiplied_clip, pos_prompt)
        neg_cond = accum_neg_cond_list[repeated_neg_txt_idx] if repeated_neg_txt else self.encode_text(multiplied_clip, neg_prompt)

        using_ollama = prompt_type == 'ollama'
        using_gguf = selected_llm and is_valid_llm
        using_original = not using_gguf and not using_ollama

        if using_gguf:
            if repeated_pos_txt:
                print(f'  - Reusing previous LLM generated prompt.\n')
                return (pos_cond, neg_cond, pos_txt, neg_txt, positive_txt, negative_txt, )
            else:
                print(f'  - Generating new LLM generated prompt.\n')
                llm_prompt_cleaned, llm_conditioning = self.load_gguf_llm(pos_prompt, multiplied_clip, max_tokens, seed, llm_model)
                tagged_pos_llm = self.tag_prompt(param_string, llm_prompt_cleaned)
                return (llm_conditioning, neg_cond, tagged_pos_llm, neg_txt, positive_txt, negative_txt, )

        if using_ollama:
            print(f'\n  - Using original prompt.\n')
            tagged_pos = self.tag_prompt(param_string, pos_txt, 'OLLAMA SUPPORT COMING SOON')
            return (pos_cond, neg_cond, tagged_pos, neg_prompt, positive_txt, negative_txt, )

        if using_original:
            print(f'\n  - Using original prompt.\n')
            warn_msg = 'NOT A VALID GGUF MODEL' if selected_llm and not is_valid_llm else ''
            tagged_pos = self.tag_prompt(param_string, pos_prompt, warn_msg)
            return (pos_cond, neg_cond, tagged_pos, neg_prompt, positive_txt, negative_txt, )

    def update_prompt_list(self, prompt_list, prompt_params):

        (multiplied_clip, pos_prompt, neg_prompt, run_prompt, seed_to_use, run_model, token_to_use,
         q, k, v, out, prompts_out_copy) = prompt_params

        prompt_parts = self.generate_prompt(
            multiplied_clip, pos_prompt, neg_prompt, run_prompt, seed_to_use, run_model,
            token_to_use, q, k, v, out, prompts_out_copy
        )
        prompt_list[0] += [prompt_parts[0]]
        prompt_list[1] += [prompt_parts[1]]
        prompt_list[2] += [prompt_parts[2]]
        prompt_list[3] += [prompt_parts[3]]
        prompt_list[4] += [prompt_parts[4]]
        prompt_list[5] += [prompt_parts[5]]

    def extract_prompt_parts(self, prompts_out, multiplied_clip, pos_prompts, neg_prompts, prompt_combinations, run_prompt, seed_to_use, run_model,
            token_to_use, q, k, v, out):

        prompts_out_copy = copy.deepcopy(prompts_out)

        pos_count = len(pos_prompts)
        neg_count = len(neg_prompts)

        equal_counts = pos_count == neg_count
        different_pos = False if all(prompt == pos_prompts[0] for prompt in pos_prompts) else True
        different_neg = False if all(prompt == neg_prompts[0] for prompt in neg_prompts) else True

        one_neg_to_many_pos = neg_count == 1 and pos_count > 1
        one_pos_to_many_neg = pos_count == 1 and neg_count > 1

        one_to_one = equal_counts and different_pos and different_neg and prompt_combinations == 'one_to_one'
        same_neg_to_many_pos = (equal_counts and different_pos and not different_neg) or one_neg_to_many_pos
        same_pos_to_many_neg = (equal_counts and different_neg and not different_pos) or one_pos_to_many_neg

        prompt_params = [
            multiplied_clip, pos_prompts[0], neg_prompts[0], run_prompt, seed_to_use, run_model,
            token_to_use, q, k, v, out, prompts_out_copy
        ]

        if one_to_one:
            for p, pos_prompt in enumerate(pos_prompts):
                prompt_params[1] = pos_prompt
                prompt_params[2] = neg_prompts[p]
                self.update_prompt_list(prompts_out_copy, prompt_params)

        elif same_neg_to_many_pos:
            for p, pos_prompt in enumerate(pos_prompts):
                prompt_params[1] = pos_prompt
                self.update_prompt_list(prompts_out_copy, prompt_params)

        elif same_pos_to_many_neg:
            for n, neg_prompt in enumerate(neg_prompts):
                prompt_params[2] = neg_prompt
                self.update_prompt_list(prompts_out_copy, prompt_params)

        else:
            for p, pos_prompt in enumerate(pos_prompts):
                for n, neg_prompt in enumerate(neg_prompts):
                    prompt_params[1] = pos_prompt
                    prompt_params[2] = neg_prompt
                    self.update_prompt_list(prompts_out_copy, prompt_params)

        return prompts_out_copy

    def generate_prompts(self, clip, positive_text_prompt, negative_text_prompt, prompt_combinations, prompt_type,
                         seed, llm_model, max_tokens, seed_str_list, clip_preset):

        print(
            f'\n\nFlowState Unified Prompt'
            f' - CLIP Attention Preset: {clip_preset}\n'
        )
        start_time = time.time()

        positive_prompts = [prompt.strip() for prompt in positive_text_prompt.split('-----')]
        negative_prompts = [prompt.strip() for prompt in negative_text_prompt.split('-----')]
        seed_list = seed_str_list.replace(' ', '').split(',')
        model_list = llm_model.replace(' ', '').split(',')
        token_list = max_tokens.replace(' ', '').split(',')

        q = self.format_num(CLIP_PRESETS[clip_preset][0], float, 1.0)
        k = self.format_num(CLIP_PRESETS[clip_preset][1], float, 1.0)
        v = self.format_num(CLIP_PRESETS[clip_preset][2], float, 1.0)
        out = self.format_num(CLIP_PRESETS[clip_preset][3], float, 1.0)

        multiplied_clip = self.multiply_clip_attn(clip, q, k, v, out)

        last_prompt = None
        prompts_out = [[], [], [], [], [], []]

        for token_num, run_token in enumerate(token_list):
            token_to_use = self.format_num(run_token, int, seed)

            for model_num, run_model in enumerate(model_list):
                prompt_type_list = ['extra_crispy', 'original'] if prompt_type == 'both' else [prompt_type]

                for prompt_num, run_prompt in enumerate(prompt_type_list):

                    if run_prompt == 'original':
                        prompts_out = self.extract_prompt_parts(
                            prompts_out, multiplied_clip, positive_prompts, negative_prompts, prompt_combinations,
                            run_prompt, 4, run_model, token_to_use, q, k, v, out
                        )
                    else:
                        for seed_num, run_seed in enumerate(seed_list):
                            seed_to_use = self.format_num(run_seed, int, seed)

                            prompts_out = self.extract_prompt_parts(
                                prompts_out, multiplied_clip, positive_prompts, negative_prompts, prompt_combinations,
                                run_prompt, seed_to_use, run_model, token_to_use, q, k, v, out
                            )

        prompt_gen_duration, prompt_gen_mins, prompt_gen_secs = get_mins_and_secs(start_time)

        print(
            f'\n\n\nFlowState Unified Prompt - Prompt generation complete.'
            f'\n  - Generation Time: {prompt_gen_mins}m {prompt_gen_secs}s'
        )

        if len(prompts_out[0]) == 1:
            return (prompts_out[0][0], prompts_out[1][0], prompts_out[2][0], prompts_out[3][0], )

        return (prompts_out[0], prompts_out[1], prompts_out[2], prompts_out[3], multiplied_clip, )


class FlowStatePromptOutput:
    CATEGORY = 'FlowState/prompt'
    DESCRIPTION = 'Show selected text output.'
    FUNCTION = 'show_prompt'
    RETURN_TYPES = STRING_OUT
    RETURN_NAMES = ('selected_prompt',)
    OUTPUT_TOOLTIPS = ('Either your original input text prompt or the prompt generated by the LLM.',)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text': ANY,
            },
            'hidden': {
                'unique_id': 'UNIQUE_ID', 'extra_pnginfo': 'EXTRA_PNGINFO',
            },
        }

    def show_prompt(self, text, unique_id=None, extra_pnginfo=None):
        print(f'  - Displaying output text.')
        in_text = copy.deepcopy(text) if isinstance(text, list) else [copy.deepcopy(text)]
        out_text = ''
        separator = '-' * 25

        for p, prompt in enumerate(in_text):
            if FS_LLM_PROMPT_TAG in prompt:
                build_text = prompt.replace(f'{FS_LLM_PROMPT_TAG}-', '')
                out_split = build_text.split('-----')
                param_str = out_split[0].replace(',', ', ').replace('llm_', '').replace(':', ': ')
                build_text = f'PROMPT {p + 1}:\n{param_str}\n{separator}\n{out_split[1]}\n\n\n'
                out_text += build_text
            else:
                out_text += f'PROMPT {p + 1}:\n{separator}\n{prompt}\n\n\n'

        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if 'workflow' in extra_pnginfo:
                workflow = extra_pnginfo['workflow']
            node = None
            if workflow and 'nodes' in workflow:
                node = next((x for x in workflow['nodes'] if str(x['id']) == unique_id), None)
            if node:
                node['widgets_values'] = [str(out_text)]

        print(f'\n\n out_text FINAL: \n{out_text} \n\n')
        return {'ui': {'text': (str(out_text),)}}


