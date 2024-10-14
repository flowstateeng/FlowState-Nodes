# Project: FlowState LLM Prompt & LLM Output
# Description: Generate an image prompt using an LLM.
# Version: 1.0.0
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# SYSTEM STATUS
##
print(f'    - Loaded LLM Prompt & Prompt Output nodes.')


##
# FS IMPORTS
##
from .FS_Types import *
from .FS_Constants import *
from .FS_Assets import *


##
# LIBRARIES
##
import os, sys
import importlib
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd

try:
    Llama = importlib.import_module('llama_cpp_cuda').Llama
except ImportError:
    Llama = importlib.import_module('llama_cpp').Llama


warnings.filterwarnings('ignore', message='clean_up_tokenization_spaces')
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning)


##
# NODES
##
class FlowStatePromptLLM:
    CATEGORY = 'FlowState/LLM'
    DESCRIPTION = 'Generate an image prompt using an LLM.'
    FUNCTION = 'generate_prompt'
    RETURN_TYPES = FS_LLM_OUT
    RETURN_NAMES = ('selected_conditioning', 'orig_conditioning', 'llm_text', 'orig_text', )
    OUTPUT_TOOLTIPS = (
        'Conditioning of the embedded original or LLM text used to guide the diffusion model.',
        'Conditioning of the embedded original text used to guide the diffusion model.',
        'Generated text prompt from the LLM.',
        'Your original text prompt.',
    )

    @classmethod
    def INPUT_TYPES(cls):
        available_models = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            gguf_files = [file for file in os.listdir(GLOBAL_MODELS_DIR) if file.endswith('.gguf')]
            available_models.extend(gguf_files)
        else:
            available_models.append('no available models')

        return {
            'required': {
                'text_prompt': STRING_PROMPT,
                'prompt_type': (['extra_crispy', 'original'],),
                'seed': SEED,
                'llm_model': (available_models,),
                'max_tokens': MAX_TOKENS,
                'clip': CLIP_IN,
                'q': FLOAT_CLIP_ATTN,
                'k': FLOAT_CLIP_ATTN,
                'v': FLOAT_CLIP_ATTN,
                'out': FLOAT_CLIP_ATTN,
            }
        }

    @classmethod
    def multiply_clip_attn(self, clip, q, k, v, out):
        multiplied = clip.clone()
        sd = multiplied.patcher.model_state_dict()

        for key in sd:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, q)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, k)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, v)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                multiplied.add_patches({key: (None,)}, 0.0, out)

        return multiplied

    @classmethod
    def encode_text(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([cond, output], )

    @classmethod
    def encode_text_flux(self, clip, clip_l, t5xxl, guidancet):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        output["guidance"] = guidance
        return ([[cond, output]], )

    @classmethod
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

    def generate_prompt(self, text_prompt, prompt_type, seed, llm_model, max_tokens, clip, q, k, v, out):
        print(f'\nFlowState LLM Prompt')

        multiplied_clip = self.multiply_clip_attn(clip, q, k, v, out)
        orig_conditioning = self.encode_text(multiplied_clip, text_prompt)

        if prompt_type == 'original':
            print(f'  - Using original prompt.')
            return (orig_conditioning, orig_conditioning, 'LLM NOT SELECTED', text_prompt)

        if prompt_type == 'extra_crispy' and not llm_model.endswith('.gguf'):
            print(f'  - Not a valid GGUF model.')
            return (orig_conditioning, orig_conditioning, 'NOT A GGUF MODEL', text_prompt)

        if prompt_type == 'extra_crispy' and llm_model.endswith('.gguf'):
            print(f'  - Using LLM generated prompt.')
            model_path = os.path.join(GLOBAL_MODELS_DIR, llm_model)
            generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 50, 'repeat_penalty': 1.2}
            loaded_model = Llama(model_path=model_path, n_gpu_layers=-1, seed=seed, verbose=False, n_ctx=2048,)

            llm_messages = self.generate_msgs(f'Generate a prompt from {text_prompt}')
            llm_prompt = loaded_model.create_chat_completion(llm_messages, **generate_kwargs)
            llm_prompt_cleaned = llm_prompt['choices'][0]['message']['content'].strip()

            llm_conditioning = self.encode_text(multiplied_clip, llm_prompt_cleaned)

            return (llm_conditioning, orig_conditioning, llm_prompt_cleaned, text_prompt)


class FlowStatePromptLLMOutput:
    CATEGORY = 'FlowState/LLM'
    DESCRIPTION = 'Show LLM generated output.'
    FUNCTION = 'show_prompt'
    RETURN_TYPES = STRING_OUT
    RETURN_NAMES = ('llm_prompt',)
    OUTPUT_TOOLTIPS = ('Generated prompt from the LLM.',)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ANY,
            },
            "hidden": {
                "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def show_prompt(self, text, unique_id=None, extra_pnginfo=None):
        print(f'  - Displaying output text.')
        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
            node = None
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [str(text)]
        return {"ui": {"text": (str(text),)}}


