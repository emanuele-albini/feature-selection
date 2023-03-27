# __all__ = ['MultivariateFilterMethod']

# from ..._utils import enum_with_attributes_factory
from ._mrmr import MRMR
from ._fcbf import FCBF
from ._cmim import CMIM

# class MultivariateFilterMethod(enum_with_attributes_factory('filter_class')):
#     MIN_REDUNDANCY_MAX_RELEVANCE = ('mrmr', MRMR)
#     FAST_CORRELATION_BASED_FILTER = ('fcbf', FCBF)
#     COND_MUTUAL_INFO_MAXIMISATION = ('cmim', CMIM)