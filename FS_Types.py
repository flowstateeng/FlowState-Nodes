import sys


##
# ANY TYPE
##
class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

ANY = (AnyType('*'), {})


##
# INPUT TYPES
##
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
LATENT_CHOOSER = ('LATENT', 'INT', 'INT', )
IMAGE = ('IMAGE', )

