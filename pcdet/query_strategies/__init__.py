from __future__ import absolute_import

from .random_sampling import RandomSampling
from .entropy_sampling import EntropySampling
from .badge_sampling import BadgeSampling
from .coreset_sampling import CoresetSampling
from .llal_sampling import LLALSampling
from .montecarlo_sampling import MonteCarloSampling
from .confidence_sampling import ConfidenceSampling
from .open_crb_sampling import OpenCRBSampling
from .NTK_sampling import NTKSampling
from .prob_sampling import ProbSampling
from .react_sampling import ReactSampling
from .gradnorm_sampling import GradnormSampling
from .cider_sampling import CiderSampling

__factory = {
    'random': RandomSampling,
    'entropy': EntropySampling,
    'badge': BadgeSampling,
    'coreset': CoresetSampling,
    'llal': LLALSampling,
    'ntk': NTKSampling,
    'montecarlo': MonteCarloSampling,
    'confidence': ConfidenceSampling,
    'open-crb': OpenCRBSampling,
    'prob': ProbSampling,
    'react': ReactSampling,
    'gradnorm': GradnormSampling,
    'cider': CiderSampling
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, cur_epoch=None):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    if method != 'open-crb':
        return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)
    else:
        return __factory[method](model, labelled_loader, unlabelled_loader,
                                 rank, active_label_dir, cfg, cur_epoch=cur_epoch)