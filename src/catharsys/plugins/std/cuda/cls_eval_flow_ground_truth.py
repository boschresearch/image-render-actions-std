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


class CEvalFlowGroundTruth:

    # ##################################################################################################################
    # Class constructor. Compiles kernel for given parameters.
    def __init__(
        self,
        *,
        _tiImageShape: tuple[int, int, int],
        _tiSearchRadiusXY: tuple[int, int],
        _tiStartXY: tuple[int, int] = (0, 0),
        _tiRangeXY: tuple[int, int] = (0, 0),
    ):
        self._kernFlow = None
        self._aIdxMapXY: np.ndarray = None
        self._aSubPixMapXY: np.ndarray = None

        if len(_tiImageShape) < 2 or len(_tiImageShape) > 3:
            raise CAnyError_Message(sMsg="Image shape must be 2 or 3 dimenional")
        # endif

        self._tiSizeXY: tuple[int, int] = (_tiImageShape[1], _tiImageShape[0])
        self._iPosChanCnt: int = None

        if len(_tiImageShape) == 3:
            self._iPosChanCnt = _tiImageShape[2]
        else:
            self._iPosChanCnt = 1
        # endif

        self._tiSearchRadiusXY = _tiSearchRadiusXY
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

        # Load the flow evaluation CUDA kernel from ressources
        try:
            xKernelFlow = res.files(catharsys.plugins.std).joinpath("res").joinpath("EvalFlowGroundTruth.cu")
            with res.as_file(xKernelFlow) as pathKernelFlow:
                sKernelFlowCode = pathKernelFlow.read_text()
            # endwith
        except Exception as xEx:
            raise CAnyError_Message(sMsg="Error loading CUDA kernel for flow ground truth evaluation", xChildEx=xEx)
        # endtry

        iPosRowStride = self._tiSizeXY[0] * self._iPosChanCnt

        self._iIdxChanCnt: int = 5
        iIdxRowStride = self._iIdxChanCnt * self._tiRangeXY[0]

        self._iSubPixChanCnt: int = 4
        iSubPixRowStride = self._iSubPixChanCnt * self._tiRangeXY[0]

        sFuncFlowExp = (
            f"EvalFlow<{self._tiStartXY[0]}, {self._tiStartXY[1]}, "
            f"{self._tiRangeXY[0]}, {self._tiRangeXY[1]}, "
            f"{self._tiSizeXY[0]}, {self._tiSizeXY[1]}, "
            f"{self._tiSearchRadiusXY[0]}, {self._tiSearchRadiusXY[1]}, "
            f"{self._iPosChanCnt}, {iPosRowStride}, "
            f"{self._iIdxChanCnt}, {iIdxRowStride}, "
            f"{self._iSubPixChanCnt}, {iSubPixRowStride}>"
        )

        try:
            modKernel = cp.RawModule(code=sKernelFlowCode, options=("-std=c++11",), name_expressions=[sFuncFlowExp])
            self._kernFlow = modKernel.get_function(sFuncFlowExp)
        except Exception as xEx:
            raise CAnyError_Message(sMsg="Error compiling flow ground truth evaluation kernel", xChildEx=xEx)
        # endtry

    # enddef

    # ##################################################################################################################
    # properties

    @property
    def aIdxMapXY(self) -> np.ndarray:
        return self._aIdxMapXY

    # enddef

    @property
    def aSubPixMapXY(self) -> np.ndarray:
        return self._aSubPixMapXY

    # enddef

    # ##################################################################################################################
    # Convert array of spherical coordiantes to cartesian coordinates
    def _SphericalToCartesian(self, _imgPos: np.ndarray, *, _tRgbIdx: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:

        imgPos = np.zeros_like(_imgPos)

        imgRad = _imgPos[:, :, _tRgbIdx[0]]
        imgTheta = _imgPos[:, :, _tRgbIdx[1]]
        imgPhi = _imgPos[:, :, _tRgbIdx[2]] - math.pi
        imgSinTheta = np.sin(imgTheta)

        imgPos[:, :, _tRgbIdx[0]] = imgRad * imgSinTheta * np.cos(imgPhi)
        imgPos[:, :, _tRgbIdx[1]] = imgRad * imgSinTheta * np.sin(imgPhi)
        imgPos[:, :, _tRgbIdx[2]] = imgRad * np.cos(imgTheta)

        return imgPos

    # enddef

    # ##################################################################################################################
    # Run the CUDA kernel to evaluate the flow ground truth
    def Eval(
        self,
        *,
        _imgPos1: np.ndarray,
        _imgPos2: np.ndarray,
        _imgObjIdx1: np.ndarray,
        _imgObjIdx2: np.ndarray,
        _iChObjIdx: int,
        _iChInstId: int,
        _bSphericalCS: bool = False,
        _tRgbIdx: tuple[int, int, int] = (0, 1, 2),
    ):
        """Evaluate the image flow ground truth from local object position and object index renders.
        The flow is calculated from image 1 to image 2. The result is stored in the class instance in two variables:
        aIdxMapXY and aSubPixMapXY, which are both flat lists of the size set by _tiRangeXY in the constructure.
        These variables have the following content:

        aIdxMapXY: (array of array of 5 integers)
            0: A unique object index.
            1: The start position x-coordinate (horizontal from left to right)
            2: The start position y-coordinate (vertical from top to bottom)
            3: The end position x-coordinate
            4: The end position y-coordinate

        aSubPixMapXY: (array of array of 4 floats)
            0: The start position x-coordinate
            1: The start position y-coordinate
            2: The sub-pixel flow vector x-coordinate from image 1 to 2
            3: The sub-pixel flow vector y-coordinate from image 1 to 2

        Parameters
        ----------
        _imgPos1 : np.ndarray, dimension (y-dim, x-dim, 3)
            The start image of the rendered local object coordinates
        _imgPos2 : np.ndarray, dimension (y-dim, x-dim, 3)
            The end image of the rendered local object coordinates
        _imgObjIdx1 : np.ndarray, dimension (y-dim, x-dim, 3)
            The start image of the rendered object indices.
            For each pixel, there are three floating point values: Object index, Material index, Instance id
        _imgObjIdx2 : np.ndarray, dimension (y-dim, x-dim, 3)
            The end image of the rendered object indices.
            For each pixel, there are three floating point values: Object index, Material index, Instance id
        _iChObjIdx : int
            The index of the channel that contains the object index in _imgObjIdx1 und _imgObjIdx2.
        _iChInstId : int
            The index of the channel that contains the instance id in _imgObjIdx1 und _imgObjIdx2.
        _bSphericalCS : bool
            If true, the _imgPos1 and _imgPos2 contain 3D coordinates in a spherical coordinate system,
            in the order [Radius, Theta (inclination), Phi (azimuth)] for color channels [Red, Green, Blue].
            See also https://en.wikipedia.org/wiki/Spherical_coordinate_system.
        _tRgbIdx : tuple[int, int, int]
            The indices of the red, green and blue channel in _imgPos1 and _imgPos2.
        Raises
        ------
        CAnyError_Message
            For any usage error.
        """

        self._aIdxMapXY: np.ndarray = None
        self._aSubPixMapXY: np.ndarray = None

        tSize2d: tuple[int, int] = _imgPos1.shape[0:2]

        if tSize2d != _imgPos2.shape[0:2] or tSize2d != _imgObjIdx1.shape[0:2] or tSize2d != _imgObjIdx2.shape[0:2]:
            raise CAnyError_Message(sMsg="Given images have different sizes")
        # endif

        iRows, iCols, iChan = _imgObjIdx1.shape

        imgObjFlat1 = _imgObjIdx1.reshape(-1, iChan)
        imgObjFlatId1 = imgObjFlat1[:, _iChInstId] + imgObjFlat1[:, _iChObjIdx]

        aU, aMaskObjIdx1 = cp.unique(imgObjFlatId1, return_inverse=True)
        imgObjUid1 = aMaskObjIdx1.reshape(iRows, iCols)

        aObjFlat2 = _imgObjIdx2.reshape(-1, iChan)
        aObjFlatId2 = aObjFlat2[:, _iChObjIdx] + aObjFlat2[:, _iChInstId]
        aObjIdxFlat2 = np.ones((iRows * iCols), dtype=int) * -1

        for iObjIdx in range(len(aU)):
            aMask = aObjFlatId2 == aU[iObjIdx].item()
            aObjIdxFlat2[aMask] = iObjIdx
        # endfor

        imgObjUid2 = aObjIdxFlat2.reshape(iRows, iCols)

        aInvalidIdx = np.argwhere(aU < 1.0)
        for iIdx in aInvalidIdx:
            imgObjUid1[imgObjUid1 == iIdx.item()] = -1
            imgObjUid2[imgObjUid2 == iIdx.item()] = -1
        # endfor

        caIdxMapXY = cp.ones((self._tiRangeXY[1], self._tiRangeXY[0], self._iIdxChanCnt), dtype=cp.int32)
        caIdxMapXY *= -1

        caSubPixMapXY = cp.full(
            (self._tiRangeXY[1], self._tiRangeXY[0], self._iSubPixChanCnt), cp.nan, dtype=cp.float32
        )

        # Convert imgPos images if needed
        if _bSphericalCS is True:
            caPos1 = cp.asarray(self._SphericalToCartesian(_imgPos1, _tRgbIdx=_tRgbIdx), dtype=cp.float32)
            caPos2 = cp.asarray(self._SphericalToCartesian(_imgPos2, _tRgbIdx=_tRgbIdx), dtype=cp.float32)
        else:
            caPos1 = cp.asarray(_imgPos1, dtype=cp.float32)
            caPos2 = cp.asarray(_imgPos2, dtype=cp.float32)
        # endif

        caObjUid1 = cp.asarray(imgObjUid1, dtype=cp.int32)
        caObjUid2 = cp.asarray(imgObjUid2, dtype=cp.int32)

        self._kernFlow(
            self._tiBlockDimXY, (self._iThreadCnt,), (caPos1, caPos2, caObjUid1, caObjUid2, caIdxMapXY, caSubPixMapXY)
        )

        self._aIdxMapXY = cp.asnumpy(caIdxMapXY)
        self._aSubPixMapXY = cp.asnumpy(caSubPixMapXY)

    # enddef

    # ##################################################################################################################
    # Write flow data to EXR file

    def SaveFlowImage(self, _xPathFile: Union[str, list, tuple, Path]):
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
        if pathFile.suffix != ".exr":
            raise CAnyError_Message(
                sMsg="Flow image can only be saved in OpenEXR format. Please provide a filename with suffix '.exr'"
            )
        # endif

        if self.aIdxMapXY is None or self.aSubPixMapXY is None:
            raise CAnyError_Message(sMsg="Flow has not been evaluated")
        # endif

        # Reversed order of color channels, due to OpenCV BGR image convention.
        iChX: int = 2
        iChY: int = 1
        iChZ: int = 0

        iPosCols, iPosRows = self._tiSizeXY

        aMaskValidFlow = self.aIdxMapXY[:, :, 3] >= 0
        aValidIdxXY = self.aIdxMapXY[aMaskValidFlow][:, 1:3]
        aValidIdxPos = aValidIdxXY[:, 1] * iPosCols + aValidIdxXY[:, 0]

        aFlowXY = self.aSubPixMapXY[aMaskValidFlow][:, 2:4]
        aObjIdx = self.aIdxMapXY[aMaskValidFlow][:, 0].astype(float)

        imgValidMap = np.zeros((iPosRows, iPosCols), dtype=bool)
        aValidMapFlat = imgValidMap.flatten()
        aValidMapFlat[aValidIdxPos] = True
        imgValidMap = aValidMapFlat.reshape(iPosRows, iPosCols)

        imgFlow = np.zeros((iPosRows, iPosCols, 4), dtype=np.float32)
        imgFlow[imgValidMap, iChX] = aFlowXY[:, 0]
        imgFlow[imgValidMap, iChY] = aFlowXY[:, 1]
        imgFlow[imgValidMap, iChZ] = aObjIdx
        imgFlow[imgValidMap, 3] = 1.0

        try:
            cv2.imwrite(pathFile.as_posix(), imgFlow)
        except Exception as xEx:
            raise CAnyError_Message(
                sMsg=f"Error writing flow ground truth image to path: {(pathFile.as_posix())}", xChildEx=xEx
            )
        # endtry

    # enddef


# endclass
