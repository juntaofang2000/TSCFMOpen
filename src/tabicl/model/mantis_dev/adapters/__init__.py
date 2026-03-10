"""Init file for adapters.""" 

from .projector import MultichannelProjector
from .var_selector import VarianceBasedSelector
from .diff_adapter import LinearChannelCombiner
from .sliding_concat import SlidingWindowChannelConcat


__all__ = ['MultichannelProjector', 'VarianceBasedSelector', 'LinearChannelCombiner', 'SlidingWindowChannelConcat']
