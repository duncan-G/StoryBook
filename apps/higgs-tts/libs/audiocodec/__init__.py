__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

from libs.audiotools import ml

ml.BaseModel.INTERN += ["dac.**"]
ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from . import utils
from .model import DAC
from .model import DACFile
