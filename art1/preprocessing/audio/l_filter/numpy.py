# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the filter function for audio signals. It provides with an infinite impulse response (IIR) or
finite impulse response (FIR) filter. This implementation is a wrapper around the `scipy.signal.lfilter` function in
the `scipy` package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

from scipy.signal import lfilter
import numpy as np
from tqdm.auto import tqdm

from art1.config import ART_NUMPY_DTYPE
from art1.preprocessing.preprocessing import Preprocessor

if TYPE_CHECKING:
    from art1.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class LFilter(Preprocessor):
    """
    This module implements the filter function for audio signals. It provides with an infinite impulse response (IIR)
    or finite impulse response (FIR) filter. This implementation is a wrapper around the `scipy.signal.lfilter`
    function in the `scipy` package.
    """

    params = ["numerator_coef", "denominator_coef", "axis", "initial_cond", "verbose"]

    def __init__(
        self,
        numerator_coef: np.ndarray = np.array([1.0]),
        denominator_coef: np.ndarray = np.array([1.0]),
        axis: int = -1,
        initial_cond: Optional[np.ndarray] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
    ):
        """
        Create an instance of LFilter.

        :param numerator_coef: The numerator coefficient vector in a 1-D sequence.
        :param denominator_coef: The denominator coefficient vector in a 1-D sequence. By simply setting the array of
                                 denominator coefficients to np.array([1.0]), this preprocessor can be used to apply a
                                 FIR filter.
        :param axis: The axis of the input data array along which to apply the linear filter. The filter is applied to
                     each subarray along this axis.
        :param initial_cond: Initial conditions for the filter delays.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.numerator_coef = numerator_coef
        self.denominator_coef = denominator_coef
        self.axis = axis
        self.initial_cond = initial_cond
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply filter to sample `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Similar samples.
        """
        x_preprocess = x.copy()

        # Filter one input at a time
        for i, x_preprocess_i in enumerate(tqdm(x_preprocess, desc="Apply audio filter", disable=not self.verbose)):
            x_preprocess[i] = lfilter(
                b=self.numerator_coef, a=self.denominator_coef, x=x_preprocess_i, axis=self.axis, zi=self.initial_cond
            )
            x_preprocess[i] = x_preprocess[i].astype(ART_NUMPY_DTYPE)

        if self.clip_values is not None:
            np.clip(x_preprocess, self.clip_values[0], self.clip_values[1], out=x_preprocess)

        return x_preprocess, y

    def _check_params(self) -> None:
        if not isinstance(self.denominator_coef, np.ndarray) or self.denominator_coef[0] == 0:
            raise ValueError("The first element of the denominator coefficient vector must be non zero.")

        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        if not isinstance(self.numerator_coef, np.ndarray):
            raise ValueError("The numerator coefficient vector has to be of type `np.ndarray`.")

        if not isinstance(self.axis, int):
            raise ValueError("The axis of the input data array has to be of type `int`.")

        if self.initial_cond is not None and not isinstance(self.initial_cond, np.ndarray):
            raise ValueError("The initial conditions for the filter delays must be of type `np.ndarray`.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
