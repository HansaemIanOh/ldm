from .ldm import LatentDiffusion
from .ddpm import DDPM
from .autoencoder import AutoEncoderKL
chose_model = {'ldm':LatentDiffusion,
               'ddpm':DDPM,
               'autoencoder':AutoEncoderKL
               }
def get_model(name, config):
    return chose_model[name](config)