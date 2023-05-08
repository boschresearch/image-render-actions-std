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

sFrame1: str = "Frame_0010.exr"
sFrame2: str = "Frame_0011.exr"


def LoadImages(_sFrame: str):
    pathImgLocalPos3d = pathLocalPos3d / _sFrame
    pathImgObjectIdx = pathObjectIdx / _sFrame

    imgLocalPos3d = cv2.imread(
        pathImgLocalPos3d.as_posix(),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
    )

    imgObjectIdx = cv2.imread(
        pathImgObjectIdx.as_posix(),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
    )

    return imgLocalPos3d, imgObjectIdx


# enddef

print("Loading images...")
imgPos1, imgObj1 = LoadImages(sFrame1)
imgPos2, imgObj2 = LoadImages(sFrame2)
print("done")

aOffset = np.array([[[1e4, 1e4, 1e4]]])
imgPos1 = imgPos1 - aOffset
imgPos2 = imgPos2 - aOffset

iChX: int = 2
iChY: int = 1
iChZ: int = 0

# ################################################################################################
# Find unique object ids

print("Finding unique objects in image 1...")
iChObjIdx = 2
iChMatIdx = 1
iChRnd = 0

iRows, iCols, iChan = imgObj1.shape

imgObjFlat1 = imgObj1.reshape(-1, iChan)
imgObjFlatId1 = imgObjFlat1[:, iChRnd] + imgObjFlat1[:, iChObjIdx]

aU, aMaskObjIdx1 = cp.unique(imgObjFlatId1, return_inverse=True)
imgMaskObjIdx1 = aMaskObjIdx1.reshape(iRows, iCols)

print("done")

print("Finding same objects in image 2...")
aObjFlat2 = imgObj2.reshape(-1, iChan)
aObjFlatId2 = aObjFlat2[:, 0] + aObjFlat2[:, 2]
aObjIdxFlat2 = np.ones((iRows * iCols), dtype=int) * -1

for iObjIdx in range(len(aU)):
    aMask = aObjFlatId2 == aU[iObjIdx].item()
    aObjIdxFlat2[aMask] = iObjIdx
# endfor

imgMaskObjIdx2 = aObjIdxFlat2.reshape(iRows, iCols)

aInvalidIdx = np.argwhere(aU < 1.0)
for iIdx in aInvalidIdx:
    imgMaskObjIdx1[imgMaskObjIdx1 == iIdx.item()] = -1
    imgMaskObjIdx2[imgMaskObjIdx2 == iIdx.item()] = -1
# endfor

print("done")

# ################################################################################################
# Load and compile CUDA kernel

print("Loading & compiling CUDA kernel...")
pathKernel = Path(__file__).parent / "Dev-EvalFlow-v3.cu"
sKernelCode = pathKernel.read_text()

iThreadCnt = 32
tiSearchRadiusXY = (200, 200)

iPosRows, iPosCols, iPosChanCnt = imgPos1.shape
tiSizeXY = (iPosCols, iPosRows)
tiStartXY = (3250, 1250)
tiRangeXY = (500, 1)

# Full image
tiStartXY = (0, 0)
tiRangeXY = (iPosCols, iPosRows)

tiRangeXY = tuple(
    tiRangeXY[i] if tiStartXY[i] + tiRangeXY[i] <= tiSizeXY[i] else tiSizeXY[i] - tiStartXY[0] for i in range(2)
)

tiBlockDimXY = (tiRangeXY[0] // iThreadCnt + (1 if tiRangeXY[0] % iThreadCnt > 0 else 0), tiRangeXY[1])

iIdxCols = tiRangeXY[0]
iIdxRows = tiRangeXY[1]
iIdxChan = 5

iPosRowStride = iPosCols * iPosChanCnt
iIdxRowStride = iIdxChan * iIdxCols

iSubPixChanCnt = 4
iSubPixRowStride = iSubPixChanCnt * iIdxCols

sFuncFlowExp = (
    f"EvalFlow<{tiStartXY[0]}, {tiStartXY[1]}, "
    f"{tiRangeXY[0]}, {tiRangeXY[1]}, "
    f"{tiSizeXY[0]}, {tiSizeXY[1]}, "
    f"{tiSearchRadiusXY[0]}, {tiSearchRadiusXY[1]}, "
    f"{iPosChanCnt}, {iPosRowStride}, "
    f"{iIdxChan}, {iIdxRowStride}, "
    f"{iSubPixChanCnt}, {iSubPixRowStride}>"
)

modKernel = cp.RawModule(code=sKernelCode, options=("-std=c++11",), name_expressions=[sFuncFlowExp])
# modKernel.compile(log_stream=sys.stdout)
kernFlow = modKernel.get_function(sFuncFlowExp)
print("done")

# ################################################################################################
# Run Flow Kernel

print("Executing CUDA kernel...")
caIdxMapXY = cp.ones((iIdxRows, iIdxCols, iIdxChan), dtype=cp.int32)
caIdxMapXY *= -1

caSubPixMapXY = cp.full((iIdxRows, iIdxCols, iSubPixChanCnt), cp.nan, dtype=cp.float32)

caPos1 = cp.asarray(imgPos1, dtype=cp.float32)
caPos2 = cp.asarray(imgPos2, dtype=cp.float32)
caMaskObjIdx1 = cp.asarray(imgMaskObjIdx1, dtype=cp.int32)
caMaskObjIdx2 = cp.asarray(imgMaskObjIdx2, dtype=cp.int32)

kernFlow(tiBlockDimXY, (iThreadCnt,), (caPos1, caPos2, caMaskObjIdx1, caMaskObjIdx2, caIdxMapXY, caSubPixMapXY))

# cp.cuda.Stream.null.synchronize()

aIdxMapXY = cp.asnumpy(caIdxMapXY)
aSubPixMapXY = cp.asnumpy(caSubPixMapXY)
print("done")

# ################################################################################################
# Saving flow image
pathFlow = pathMain / "Flow.exr"
print(f"Writing flow image to: {pathFlow}")

aMaskValidFlow = aIdxMapXY[:, :, 3] >= 0
aValidIdxXY = aIdxMapXY[aMaskValidFlow][:, 1:3]
aValidIdxPos = aValidIdxXY[:, 1] * iPosCols + aValidIdxXY[:, 0]

aFlowXY = aSubPixMapXY[aMaskValidFlow][:, 2:4]
aObjIdx = aIdxMapXY[aMaskValidFlow][:, 0].astype(float)

imgValidMap = np.zeros((iPosRows, iPosCols), dtype=bool)
aValidMapFlat = imgValidMap.flatten()
aValidMapFlat[aValidIdxPos] = True
imgValidMap = aValidMapFlat.reshape(iPosRows, iPosCols)

imgFlow = np.zeros((iPosRows, iPosCols, 4), dtype=np.float32)
imgFlow[imgValidMap, iChX] = aFlowXY[:, 0]
imgFlow[imgValidMap, iChY] = aFlowXY[:, 1]
imgFlow[imgValidMap, iChZ] = aObjIdx
imgFlow[imgValidMap, 3] = 1.0

cv2.imwrite(pathFlow.as_posix(), imgFlow)

# ################################################################################################
# Get flat list of start and mapped pixel indices
iSelObjIdx = 2
print(f"Plotting flow image for object id {iSelObjIdx}...")

# Select object id and those points that have a valid map
aMaskObjIdx = np.logical_and(aIdxMapXY[:, :, 0] == iSelObjIdx, aIdxMapXY[:, :, 3] >= 0)

aFlatIdxXY = aIdxMapXY[aMaskObjIdx][:, 1:3].astype(float)
aFlatMapIdxXY = aSubPixMapXY[aMaskObjIdx][:, 0:2]
aFlowLines = np.stack((aFlatIdxXY, aFlatMapIdxXY), axis=1)

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
