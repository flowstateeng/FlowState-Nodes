# Project: ComfyUI FlowState Nodes
# Description: A collection of custom nodes to solve problems I couldn't find existing nodes for.
# Version: 0.0.1
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# IMPORTS
##
import torch
import numpy as np

import os, sys
import hashlib
import nodes
import node_helpers
import folder_paths
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd

from PIL import Image, ImageOps, ImageSequence
from comfy import model_management
from comfy.utils import load_torch_file
from nodes import UNETLoader
from nodes import CheckpointLoaderSimple
from custom_nodes.ComfyUI_bitsandbytes_NF4 import CheckpointLoaderNF4


try:
    Llama = importlib.import_module('llama_cpp_cuda').Llama
except ImportError:
    Llama = importlib.import_module('llama_cpp').Llama


##
# ASSETS
##
WEB_DIRECTORY = './web'


##
# INPUT TYPES
##
class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

ANY = (AnyType('*'), {})

FLOAT = ('FLOAT', { 'default': 1, 'min': -sys.float_info.max, 'max': sys.float_info.max, 'step': 0.01 })
FLOAT_CLIP = ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01})
INT = ('INT', { 'default': 1, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1 })
SEED = ('INT', { 'default': 4, 'min': -sys.maxsize, 'max': sys.maxsize, 'step': 1 })
MAX_TOKENS = ('INT', {'default': 4096, 'min': 1, 'max': 8192})

BOOLEAN = ('BOOLEAN', {'default': True})
BOOLEAN_FALSE = ('BOOLEAN', {'default': False})
BOOLEAN_TRUE = ('BOOLEAN', {'default': True})

STRING_IN = ('STRING', {'default': 'Enter a value.'})
STRING_IN_FORCED = ('STRING', {'forceInput': True})
STRING_ML = ('STRING', {'multiline': True, 'default': 'Enter a value.'})
STRING_PROMPT = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'A stunning image.'})
STRING_PROMPT_LLM = ('STRING', {'multiline': True, 'dynamicPrompts': True, 'default': 'Generated LLM prompt will show here.'})
STRING_WIDGET = ('STRING', {'forceInput': True})

CLIP_IN = ('CLIP', {'tooltip': 'The CLIP model used for encoding the text.'})

JSON_WIDGET = ('JSON', {'forceInput': True})
METADATA_RAW = ('METADATA_RAW', {'forceInput': True})


##
# OUTPUT TYPES
##
MODEL = ('MODEL', )
MODEL_UNIFIED = ('MODEL', 'CLIP', 'VAE', )
CONDITIONING = ('CONDITIONING', )

STRING_OUT = ('STRING', )
STRING_OUT_2 = ('STRING', 'STRING', )
FS_LLM_OUT = ('CONDITIONING', 'CONDITIONING', 'STRING', 'STRING', )

LATENT = ('LATENT', )
IMAGE = ('IMAGE', )


##
# CONSTANTS
##
MAX_RESOLUTION=16384
GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, 'llm_gguf')
DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'


##
# NODES
##
class FlowStateUnifiedModelLoader:
    CATEGORY = 'FlowState/loader'
    DESCRIPTION = 'Load checkpoints and UNETs, includes NF4 support.'
    FUNCTION = 'load_model'
    RETURN_TYPES = MODEL_UNIFIED
    RETURN_NAMES = ('model', 'clip', 'vae', )
    OUTPUT_TOOLTIPS = (
        'Checkpoint or UNET model.',
        'The CLIP model used for encoding text prompts.',
        'The VAE model used for encoding and decoding images to and from latent space.',
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'nf4_name': (folder_paths.get_filename_list('checkpoints'), ),
                'ckpt_name': (folder_paths.get_filename_list('checkpoints'), {'tooltip': 'Uses included CLIP & VAE'}, ),
                'unet_name': (folder_paths.get_filename_list('diffusion_models'), ),
                'weight_dtype': (['default', 'fp8_e4m3fn', 'fp8_e5m2'], ),
                'model_type': (['ckpt', 'unet', 'nf4'],),
                'clip_1': (folder_paths.get_filename_list('clip'), ),
                'clip_2': (folder_paths.get_filename_list('clip'), ),
                'clip_type': (['default', 'sdxl', 'sd3', 'flux'], ),
                "vae_name": (['default'] + s.vae_list(), ),
            }
        }

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return vae

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path_or_raise("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("clip", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)
        return clip

    def load_model(self, nf4_name, ckpt_name, unet_name, weight_dtype, model_type, clip_1, clip_2, clip_type, vae_name):
        model, clip, vae = None, None, None

        if model_type == 'nf4':
            print(f' FlowState Unified Loader - loading: {nf4_name}')
            loader = CheckpointLoaderNF4()
            loaded_model = loader.load_checkpoint(nf4_name)
            model = loaded_model[0]
            if clip_type == 'default': clip = loaded_model[1]
            if vae_name == 'default': vae = loaded_model[2]

        if model_type == 'ckpt':
            print(f' FlowState Unified Loader - loading: {ckpt_name}')
            loader = CheckpointLoaderSimple()
            loaded_model = loader.load_checkpoint(ckpt_name)
            model = loaded_model[0]
            if clip_type == 'default': clip = loaded_model[1]
            if vae_name == 'default': vae = loaded_model[2]

        if model_type == 'unet':
            print(f' FlowState Unified Loader - loading: {unet_name}')
            loader = UNETLoader()
            model = loader.load_unet(unet_name, weight_dtype)[0]

        if vae == None:
            vae = self.vae_list()[0] if vae_name == 'default' else self.load_vae(vae_name)

        if clip == None and clip_type != 'default':
            clip = self.load_clip(clip_1, clip_2, clip_type)

        if clip == None: clip = self.load_clip(clip_1, clip_2, 'flux')

        return (model, clip, vae,)


class FlowStateLatentChooser:
    CATEGORY = 'FlowState/latent'
    DESCRIPTION = 'Create a new batch of latent images to be denoised via sampling.'
    FUNCTION = 'create_latent'
    RETURN_TYPES = LATENT
    OUTPUT_TOOLTIPS = ('The latent image batch.',)

    @classmethod
    def __init__(self):
        self.device = model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            'required': {
                'width': ('INT', {'default': 512, 'min': 16, 'max': MAX_RESOLUTION, 'step': 8, 'tooltip': 'The width of the latent images in pixels.'}),
                'height': ('INT', {'default': 512, 'min': 16, 'max': MAX_RESOLUTION, 'step': 8, 'tooltip': 'The height of the latent images in pixels.'}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 4096, 'tooltip': 'The number of latent images in the batch.'}),
                'latent_type': (['empty_latent', 'input_img', 'imported_img'],),
                'image': (sorted(files), {'image_upload': True}),
                'vae': ('VAE', )
            },
            'optional': {
                'pixels': IMAGE
            }
        }

    @classmethod
    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({'samples':latent}, )

    @classmethod
    def VALIDATE_INPUTS(s, image, vae=None):
        if not folder_paths.exists_annotated_filepath(image):
            return 'Invalid image file: {}'.format(image)
        return True

    @classmethod
    def load_and_encode(self, image, vae):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            image = i.convert('RGB')

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device='cpu')
            output_images.append(image)

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        encoded = vae.encode(output_image[:,:,:,:3])
        return ({'samples': encoded}, )

    def create_latent(self, latent_type, image, vae, width, height, batch_size=1, pixels=None):
        if latent_type == 'empty_latent':
            return self.generate(width, height, batch_size)
        elif latent_type == 'input_img':
            return ({'samples': vae.encode(pixels[:,:,:,:3])}, )
        else:
            return self.load_and_encode(image, vae)


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
                'model': (available_models,),
                'max_tokens': MAX_TOKENS,
                'clip': CLIP_IN,
                'q': FLOAT_CLIP,
                'k': FLOAT_CLIP,
                'v': FLOAT_CLIP,
                'out': FLOAT_CLIP,
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

    def generate_prompt(self, text_prompt, prompt_type, seed, model, max_tokens, clip, q, k, v, out):
        multiplied_clip = self.multiply_clip_attn(clip, q, k, v, out)
        orig_conditioning = self.encode_text(multiplied_clip, text_prompt)

        if prompt_type == 'original':
            return (orig_conditioning, orig_conditioning, 'LLM NOT SELECTED', text_prompt)

        if prompt_type == 'extra_crispy' and not model.endswith('.gguf'):
            return (orig_conditioning, orig_conditioning, 'NOT A GGUF MODEL', text_prompt)

        if prompt_type == 'extra_crispy' and model.endswith('.gguf'):
            model_path = os.path.join(GLOBAL_MODELS_DIR, model)
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
