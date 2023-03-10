"""
    This module provides the wrappers for ML models implementing the standard interface (scikit-learn interface)
"""

__author__ = 'Emanuele Albini'
__all__ = [
    'SKLWrapper',
    'UniversalSKLWrapper',
]

import warnings


class SKLWrapper(object):
    """Inference Wrapper with a standatd SKL (scikit-learn).

        Usage:
            new_model = SKLWrapper(model, *args, **kwargs)
            new_model will be a wrapper or monkey-patched version of the original model.
        
        It will implement:
        - predict : That returns the prediction
        - predict_proba : That returns the probabilities of each class
        - decision_function (optional) : That return the raw results of the decision function

    """
    def __new__(cls, model, *args, **kwargs):
        """Wraps or Monkey-patch an existing model

        Args:
            model : The model
        
        Based on the type of model different arguments will be available.
        Based on the type of model a different model will be returned.
        
        - xgboost.Booster : An XGBClassifierSKLWrapper will be returned.
                            The arguments will be passed to the constructor of XGBClassifierSKLWrapper.
                            See XGBClassifierSKLWrapper for more details on the available arguments.
        - skleran.*       : The class will be monkey-patched to implement the missing methods.

        """
        module_name, class_name = model.__class__.__module__, model.__class__.__name__

        # XGBoost Booster (binary)
        if (module_name, class_name) == ('xgboost.core', 'Booster'):
            # We import locally so that we don't need to have XGBoost installed
            from .xgboost import XGBClassifierSKLWrapper
            return XGBClassifierSKLWrapper(model, *args, **kwargs)

        # SKL Classifiers (binary)
        elif module_name.startswith('sklearn') and hasattr(model, 'predict_proba'):
            # Get the number of classes
            if hasattr(model, 'n_classes_'):
                nclasses = model.n_classes_
            elif hasattr(model, 'classes_'):
                nclasses = len(model.classes_)
            else:
                nclasses = None

            if nclasses == 2:
                return skl_binary_wrapper(model, *args, **kwargs)
            else:
                raise NotImplementedError('This SKL model class is not supported.')

        else:
            raise NotImplementedError('Model type not supported.')


def skl_predict_binary(self, X, *args, **kwargs):
    return 1 * (self.predict_proba(X, *args, **kwargs)[:, 1] > self.threshold)


# Monkey patch-er for SKL binary prediction
def skl_binary_wrapper(model, threshold=.5, **kwargs):
    if len(kwargs) > 0:
        warnings.warn(f'Ignoring unsupported kwargs {list(kwargs.values())}')

    # Add attribute
    model.threshold = threshold

    # Monkey-patch predict
    monkey_patch_object_method(model, 'predict', skl_predict_binary)

    return model


UniversalSKLWrapper = SKLWrapper
