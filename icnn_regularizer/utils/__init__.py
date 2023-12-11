"""
Collection of useful ODL/torch utilities
"""
__all__ = []

from .odl_regularizers import tv_denoise, tgv_denoise
__all__ += ['tv_denoise', 'tgv_denoise']

from .odl_subsampling import Subsampling
__all__ += ['Subsampling']

from .odl_torch_wrapper import OperatorModule, OperatorFunction
__all__ += ['OperatorModule', 'OperatorFunction']

from .odl_plotting import get_image_data, plot, sample_img
__all__ += ['get_image_data', 'plot', 'sample_img']

from .torch_convex_model import BaseConvexRegularizer
__all__ += ['BaseConvexRegularizer']

from .torch_opt_routines import estimate_fista_maxiters, fista, conjugate_gradient
__all__ += ['estimate_fista_maxiters', 'fista', 'conjugate_gradient']

from .odl_dataset import AbstractDataset, DenoisingDataset1D, DenoisingDataset2D, SuperresolutionDataset2D, numpy_to_tensor
__all__ += ['AbstractDataset', 'DenoisingDataset1D', 'DenoisingDataset2D', 'SuperresolutionDataset2D', 'numpy_to_tensor']

from .odl_pdhg import pdhg
__all__ += ['pdhg']

import json


def save_dict(mydict: dict, outfile: str) -> None:
    """
    Save a dictionary as a json file
    """
    with open(outfile, 'w') as ofile:
        json.dump(mydict, ofile, indent=4, sort_keys=True)
    return


def read_json(infile: str) -> dict:
    """
    Read an input json file to a dictionary
    """
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict


__all__ += ['save_dict', 'read_json']