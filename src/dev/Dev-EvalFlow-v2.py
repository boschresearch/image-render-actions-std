#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \Dev-EvalFlow-v2.py
# Created Date: Tuesday, March 21st 2023, 11:35:05 am
# Author: Christian Perwass (CR/AEC5)
# <LICENSE id="All-Rights-Reserved">
# Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries
# </LICENSE>
###


try:
    print("Importing CUPY...")
    import cupy as cp

    print("done")
except Exception as xEx:
    print(
        "CUDA python module 'cupy' could not be imported.\n"
        "Make sure you have the NVIDIA CUDA toolkit installed and\n"
        "the 'cupy' module. See 'https://cupy.dev/' for more information.\n"
        "Note that if the 'pip' install does not work, try the 'conda' install option.\n\n"
        f"Exception reported:\n{(str(xEx))}"
    )
# endtry

# ################################################################################################
# Load test images

import os
import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


pathMain = Path(r"[main path]")
pathLocalPos3d = pathMain / "AT_LocalPos3d_Raw"
pathObjectIdx = pathMain / "AT_ObjectIdx_Raw"
pathObjectLoc3d = pathMain / "AT_ObjectLoc3d_Raw"
pathLabel = pathMain / "AT_Label/full_res/SemSeg"

sFrame1: str = "Frame_0010.exr"
sFrame2: str = "Frame_0011.exr"


def LoadImages(_sFrame: str):
    pathImgLocalPos3d = pathLocalPos3d / _sFrame
    pathImgObjectIdx = pathObjectIdx / _sFrame
    pathImgObjectLoc3d = pathObjectLoc3d / _sFrame

    imgLocalPos3d = cv2.imread(
        pathImgLocalPos3d.as_posix(),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
    )

    imgObjectIdx = cv2.imread(
        pathImgObjectIdx.as_posix(),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
    )

    imgObjectLoc3d = cv2.imread(
        pathImgObjectLoc3d.as_posix(),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
    )

    return imgLocalPos3d, imgObjectIdx, imgObjectLoc3d


# enddef

print("Loading images...")
imgPos1, imgObj1, imgLoc1 = LoadImages(sFrame1)
imgPos2, imgObj2, imgLoc2 = LoadImages(sFrame2)
print("done")

aOffset = np.array([[[1e4, 1e4, 1e4]]])
imgPos1 = imgPos1 - aOffset
imgPos2 = imgPos2 - aOffset

imgLoc1 = imgLoc1 - aOffset
imgLoc2 = imgLoc2 - aOffset

iChX: int = 2
iChY: int = 1
iChZ: int = 0

# ################################################################################################
# Find unique object ids

print("Finding unique objects in image 1...")
iRows, iCols, iChan = imgObj1.shape
imgObjFlat1 = imgObj1.reshape(-1, iChan)
aU, aMaskObjIdx1 = np.unique(imgObjFlat1, axis=0, return_inverse=True)
imgMaskObjIdx1 = aMaskObjIdx1.reshape(iRows, iCols)
print("done")

print("Finding unique objects in image 2...")
imgObjFlat2 = imgObj2.reshape(-1, iChan)
aU2, aMaskObjIdx2 = np.unique(imgObjFlat2, axis=0, return_inverse=True)
imgMaskObjIdx2 = aMaskObjIdx2.reshape(iRows, iCols)
print("done")

# ################################################################################################
# Load and compile CUDA kernel

print("Loading & compiling CUDA kernel...")
pathKernel = Path(__file__).parent / "Dev-EvalFlow-v2.cu"
sKernelCode = pathKernel.read_text()

iThreadCnt = 32
tiSearchRadiusXY = (100, 100)

iPosRows, iPosCols, iPosChanCnt = imgPos1.shape
tiSizeXY = (iPosCols, iPosRows)
tiStartXY = (3250, 1250)
tiRangeXY = (500, 500)

# Full image
# tiStartXY = (0, 0)
# tiRangeXY = (iPosCols, iPosRows)

tiRangeXY = tuple(
    tiRangeXY[i] if tiStartXY[i] + tiRangeXY[i] <= tiSizeXY[i] else tiSizeXY[i] - tiStartXY[0] for i in range(2)
)

tiBlockDimXY = (tiRangeXY[0] // iThreadCnt + (1 if tiRangeXY[0] % iThreadCnt > 0 else 0), tiRangeXY[1])

iIdxCols = tiRangeXY[0]
iIdxRows = tiRangeXY[1]
iIdxChan = 6

iPosRowStride = iPosCols * iPosChanCnt
iIdxRowStride = iIdxChan * iIdxCols

sFuncFlowExp = (
    f"EvalFlow<{tiStartXY[0]}, {tiStartXY[1]}, "
    f"{tiSizeXY[0]}, {tiSizeXY[1]}, "
    f"{tiSearchRadiusXY[0]}, {tiSearchRadiusXY[1]}, "
    f"{iPosChanCnt}, {iPosRowStride}, {iIdxRowStride}>"
)

modKernel = cp.RawModule(code=sKernelCode, options=("-std=c++11",), name_expressions=[sFuncFlowExp])
# modKernel.compile(log_stream=sys.stdout)
kernFlow = modKernel.get_function(sFuncFlowExp)
print("done")

# ################################################################################################
# Run Flow Kernel

print("Executing CUDA kernel...")
caIdxMapXY = cp.zeros((iIdxRows, iIdxCols, 3, 2), dtype=cp.int32)
caPos1 = cp.asarray(imgPos1, dtype=cp.float32)
caPos2 = cp.asarray(imgPos2, dtype=cp.float32)
caMaskObjIdx1 = cp.asarray(imgMaskObjIdx1, dtype=cp.int32)
caMaskObjIdx2 = cp.asarray(imgMaskObjIdx2, dtype=cp.int32)

kernFlow(tiBlockDimXY, (iThreadCnt,), (caPos1, caPos2, caMaskObjIdx1, caMaskObjIdx2, caIdxMapXY))
# cp.cuda.Stream.null.synchronize()

aIdxMapXY = cp.asnumpy(caIdxMapXY)
print("done")

# ################################################################################################
# Get flat list of start and mapped pixel indices
print("Plotting flow image...")
# Object indices
imgObjIdx = imgMaskObjIdx1[tiStartXY[1] : (tiStartXY[1] + tiRangeXY[1]), tiStartXY[0] : (tiStartXY[0] + tiRangeXY[0])]
aFlatObjIdx = imgObjIdx.flatten()

# Start flow indices
aIdxX = np.linspace(tiStartXY[0], tiStartXY[0] + tiRangeXY[0] - 1, tiRangeXY[0])
aIdxY = np.linspace(tiStartXY[1], tiStartXY[1] + tiRangeXY[1] - 1, tiRangeXY[1])
aMeshX, aMeshY = np.meshgrid(aIdxX, aIdxY)
aIdxXY = np.concatenate((aMeshX[:, :, np.newaxis], aMeshY[:, :, np.newaxis]), axis=2)

aFlatIdxXY = aIdxXY.reshape(-1, 2)

# Mapped flow indices
aFlatIdxMapXY = aIdxMapXY[:, :, 0, :].reshape(-1, 2)

# Mask for valid pixel maps
aMask = np.all(aFlatIdxMapXY >= 0, axis=1)

# ################################################################################################
# Plot result for selected object index

iSelObjIdx = 0
aMaskObjIdx = np.logical_and(aFlatObjIdx == iSelObjIdx, aMask)
aFlowLines = np.stack((aFlatIdxXY[aMaskObjIdx], aFlatIdxMapXY[aMaskObjIdx]), axis=1)

# Plot flow image
clnFlow = LineCollection(aFlowLines[::100], linewidths=1.0, cmap="jet", alpha=0.5)

figFlow, axFlow = plt.subplots()
# axFlow.imshow(imgObjPos1, alpha=0.5)
# axFlow.imshow(imgObjPos2, alpha=0.5)

axFlow.add_collection(clnFlow)
axFlow.set_xlim(
    max(0, tiStartXY[0] - tiSearchRadiusXY[0]), min(tiStartXY[0] + tiRangeXY[0] + tiSearchRadiusXY[0], tiSizeXY[0])
)
axFlow.set_ylim(
    min(tiStartXY[1] + tiRangeXY[1] + tiSearchRadiusXY[1], tiSizeXY[1]), max(0, tiStartXY[1] - tiSearchRadiusXY[1])
)
axFlow.margins(0.1)
print("done")
plt.show()
