#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cls_eval_flow_gt.py
# Created Date: Thursday, March 23rd 2023, 3:54:02 pm
# Author: Christian Perwass
# <LICENSE id="Apache-2.0">
#
#   Image-Render Standard Actions module
#   Copyright 2022 Robert Bosch GmbH and its subsidiaries
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# </LICENSE>
###

import os
import sys
import math

if sys.version_info < (3, 10):
    import importlib_resources as res
else:
    from importlib import resources as res
# endif

from typing import Union
from pathlib import Path

from anybase import path as anypath
from anybase.cls_any_error import CAnyError_Message
import catharsys.plugins.std

try:
    import cupy as cp
except Exception as xEx:
    raise CAnyError_Message(
        sMsg="CUDA python module 'cupy' could not be imported.\n"
        "Make sure you have the NVIDIA CUDA toolkit installed and\n"
        "the 'cupy' module. See 'https://cupy.dev/' for more information.\n"
        "Note that if the 'pip' install does not work, try the 'conda' install option.\n\n"
        f"Exception reported:\n{(str(xEx))}",
        xChildEx=xEx,
    )
# endtry

import numpy as np

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


class CEvalMotionBlur:

    # ##################################################################################################################
    # Class constructor. Compiles kernel for given parameters.
    def __init__(
        self,
        *,
        _tiImageShape: Union[tuple[int, int], tuple[int, int, int]],
        _tiFilterRadiusXY: tuple[int, int],
        _tiStartXY: tuple[int, int] = (0, 0),
        _tiRangeXY: tuple[int, int] = (0, 0),
    ):
        self._kernBlur = None

        if len(_tiImageShape) < 2 or len(_tiImageShape) > 3:
            raise CAnyError_Message(sMsg="Image shape must be 2 or 3 dimenional")
        # endif

        self._tiSizeXY: tuple[int, int] = (_tiImageShape[1], _tiImageShape[0])
        self._iImgChanCnt: int = None

        if len(_tiImageShape) == 3:
            self._iImgChanCnt = _tiImageShape[2]
        else:
            self._iImgChanCnt = 1
        # endif

        self._tiFilterRadiusXY = _tiFilterRadiusXY
        self._tiStartXY = _tiStartXY

        self._tiRangeXY = tuple(_tiRangeXY[i] if _tiRangeXY[i] > 0 else self._tiSizeXY[i] for i in range(2))
        self._tiRangeXY = tuple(
            self._tiRangeXY[i]
            if self._tiStartXY[i] + self._tiRangeXY[i] <= self._tiSizeXY[i]
            else self._tiSizeXY[i] - self._tiStartXY[0]
            for i in range(2)
        )

        self._iThreadCnt = 32
        self._tiBlockDimXY = (
            self._tiRangeXY[0] // self._iThreadCnt + (1 if self._tiRangeXY[0] % self._iThreadCnt > 0 else 0),
            self._tiRangeXY[1],
        )

        sKernelFlowCode: str = None
        self._aImageBlur = None

        # Load the flow evaluation CUDA kernel from ressources
        try:
            xKernelFlow = res.files(catharsys.plugins.std).joinpath("res").joinpath("EvalMotionBlur.cu")
            with res.as_file(xKernelFlow) as pathKernelFlow:
                sKernelFlowCode = pathKernelFlow.read_text()
            # endwith
        except Exception as xEx:
            raise CAnyError_Message(sMsg="Error loading CUDA kernel for motion blur evaluation", xChildEx=xEx)
        # endtry

        iImgRowStride = self._tiSizeXY[0] * self._iImgChanCnt

        self._iFlowChanCnt: int = 4
        iFlowRowStride = self._iFlowChanCnt * self._tiRangeXY[0]

        sFuncBlurExp = (
            f"EvalMotionBlur<{self._tiStartXY[0]}, {self._tiStartXY[1]}, "
            f"{self._tiRangeXY[0]}, {self._tiRangeXY[1]}, "
            f"{self._tiSizeXY[0]}, {self._tiSizeXY[1]}, "
            f"{self._tiFilterRadiusXY[0]}, {self._tiFilterRadiusXY[1]}, "
            f"{iImgRowStride}, {iFlowRowStride}, "
            f"{self._iImgChanCnt}, {self._iFlowChanCnt}>"
        )

        try:
            modKernel = cp.RawModule(code=sKernelFlowCode, options=("-std=c++11",), name_expressions=[sFuncBlurExp])
            self._kernBlur = modKernel.get_function(sFuncBlurExp)
        except Exception as xEx:
            raise CAnyError_Message(sMsg="Error compiling motion blur evaluation kernel", xChildEx=xEx)
        # endtry

    # enddef

    # ##################################################################################################################
    # properties

    @property
    def aImageBlur(self) -> np.ndarray:
        return self._aImageBlur

    # enddef

    # ##################################################################################################################
    # Run the CUDA kernel to evaluate the flow ground truth
    def Eval(
        self,
        *,
        _imgImage1: np.ndarray,
        _imgImage2: np.ndarray,
        _imgFlow: np.ndarray,
    ):
        """Evaluate motion blur between two images using their optical flow."""

        caImage1 = cp.asarray(_imgImage1, dtype=cp.float32)
        caImage2 = cp.asarray(_imgImage2, dtype=cp.float32)
        caFlow = cp.asarray(_imgFlow, dtype=cp.float32)
        caResult = cp.full((self._tiSizeXY[1], self._tiSizeXY[0], self._iImgChanCnt), 0.0, dtype=cp.float32)

        fFlowFactor: float = 1.0
        cfFlowFactor = cp.float32(fFlowFactor)

        self._kernBlur(self._tiBlockDimXY, (self._iThreadCnt,), (caImage1, caImage2, caFlow, cfFlowFactor, caResult))

        self._aImageBlur = cp.asnumpy(caResult)

    # enddef

    # ##################################################################################################################
    # Write flow data to EXR file

    def SaveBlurImage(self, _xPathFile: Union[str, list, tuple, Path]):
        """Save flow image to file.
           The filename must end in '.exr', to store the image in OpenEXR format.

        Parameters
        ----------
        _xPathFile : Union[str, list, tuple, Path]
            The filename with path to store the image. If a list or tuple is passed,
            the elements are concatenated as path elements into a full path.

            The resultant image has four float-32bit channels:
                0: The flow vector x-coordinate
                1: The flow vector y-coordinate
                2: A unique object index.
                3: Valid flag, 1 for valid flow vectors, 0 for pixels where no flow could be calculated.
        """
        pathFile = anypath.MakeNormPath(_xPathFile)

        if self._aImageBlur is None:
            raise CAnyError_Message(sMsg="Motion Blur has not been evaluated")
        # endif

        aImgBlurWrite = self._aImageBlur[:, :, [2, 1, 0]]

        try:
            cv2.imwrite(pathFile.as_posix(), aImgBlurWrite)
        except Exception as xEx:
            raise CAnyError_Message(
                sMsg=f"Error writing motion blur image to path: {(pathFile.as_posix())}", xChildEx=xEx
            )
        # endtry

    # enddef


# endclass
