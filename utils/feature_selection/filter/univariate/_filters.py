__all__ = ['UnivariateFilterMethod']

from enum import Enum
from ..._utils import enum_with_attributes_factory
from ._others import (
    chi2,
    fisher,
    kruskal,
    pearson,
    spearman,
    entropy,
    mutual_info,
    symmetric_uncertainty,
)
from ._relief import reliefF


class UnivariateFilterMethod(enum_with_attributes_factory('scoring_function', 'require_types')):
    """ Univariate filter methods.

    Attributes
    ----------
    score : callable
        The score function.
    
    require_types : bool
        Whether the score function requires the types of the features and the target.
        If True, the score function will be called with the following signature:
        score(
            X, y, 
            discrete_features : Union[bool, Iterable[bool]] = discrete_features, 
            discrete_target : bool = discrete_target
        )
    """

    # Statistical tests
    CHI2 = ('chi2', chi2, False)
    FISHER_F_TEST = ('fisher', fisher, False)
    KRUSKAL_WALLIS_H_TEST = ('kruskal', kruskal, False)
    PEARSON = ('pearson', pearson, False)
    SPEARMAN = ('spearman', spearman, False)
    # Information theory
    ENTROPY = ('entropy', entropy, True)
    MUTUAL_INFO = ('mi', mutual_info, True)
    SYMMETRIC_UNCERTAINTY = ('su', symmetric_uncertainty, True)
    # RELIEF
    RELIEF_F = ('relief-f', reliefF, True)
