#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \imgproc.py
# Created Date: Tuesday, January 18th 2021
# Author: Dirk Fortmeier (BEG/ESD1)
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
from pathlib import Path

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np

##########################################################################
def LoadImageExr(*, sFpImage, bAsUint=False, bNormalize=True):

    imgSrc_f = cv2.imread(
        sFpImage, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
    )
    if len(imgSrc_f.shape) == 2:
        iSrcRows, iSrcCols = imgSrc_f.shape
        iSrcChnl = 1
    elif len(imgSrc_f.shape) == 3:
        iSrcRows, iSrcCols, iSrcChnl = imgSrc_f.shape
    else:
        raise RuntimeError(f"Invalid shape '{imgSrc_f.shape}' of image: {sFpImage}")
    # endif

    # Prepare image
    aIsNegInf = np.isneginf(imgSrc_f)
    aIsPosInf = np.isposinf(imgSrc_f)
    aIsNan = np.isnan(imgSrc_f)
    aMask = np.logical_or(aIsNegInf, aIsPosInf)
    aMask = np.logical_or(aMask, aIsNan)
    aMask = np.logical_not(aMask)

    fMin = imgSrc_f.min(where=aMask, initial=np.finfo("float32").max)
    fMax = imgSrc_f.max(where=aMask, initial=np.finfo("float32").min)
    fRange = fMax - fMin

    imgSrc_f[aIsPosInf] = fMax
    imgSrc_f[aIsNegInf] = fMin
    imgSrc_f[aIsNan] = fMin

    if abs(fRange) > 1e-6 and bNormalize:
        imgTrg_f = (imgSrc_f - fMin) / fRange
    else:
        imgTrg_f = imgSrc_f
    # endif

    if bAsUint:
        imgTrg_u = np.around(imgTrg_f * 255.0).astype(np.uint8)
        return imgTrg_u
    else:
        return imgTrg_f
    # endif


# enddef


##########################################################################
def CreateVideo(*, lFpImages, iFps=10):

    pathImg = Path(lFpImages[0])
    sFpVideo = os.path.normpath(os.path.join(pathImg.parent.as_posix(), "Video.mp4"))

    # Find maximal extend of images in list
    iMaxW = 0
    iMaxH = 0
    for sFpImage in lFpImages:
        imgX = cv2.imread(sFpImage)
        iH, iW, iCh = imgX.shape
        iMaxW = max(iMaxW, iW)
        iMaxH = max(iMaxH, iH)
    # endfor

    tSize = (iMaxW, iMaxH)

    # print(tSize)

    xWriter = cv2.VideoWriter(sFpVideo, cv2.VideoWriter_fourcc(*"MP4V"), iFps, tSize)
    for sFpImg in lFpImages:
        # print(sFpImg)
        if sFpImg.endswith(".exr"):
            imgX = LoadImageExr(sFpImage=sFpImg, bAsUint=True)
        else:
            imgX = cv2.imread(sFpImg).astype(np.uint8)
        # endif

        iH, iW, iCh = imgX.shape
        imgMax = np.zeros((iMaxH, iMaxW, iCh), np.uint8)
        imgMax[0:iH, 0:iW, :] = imgX

        xWriter.write(imgMax)
    # endfor
    xWriter.release()

    return sFpVideo


# enddef


##########################################################################
def UpdateWidthHeight(_iImgWidth, _iImgHeight, aImage):

    iH, iW = aImage.shape[:2]
    fAspect = iH / iW

    if _iImgHeight is None:
        iImgHeight = 0
    else:
        iImgHeight = _iImgHeight
    # endif

    if _iImgWidth is None:
        iImgWidth = 0
    else:
        iImgWidth = _iImgWidth
    # endif

    if iImgWidth > 0 and iImgHeight <= 0:
        iImgHeight = max(1, int(iImgWidth * fAspect))
    elif iImgHeight > 0 and iImgWidth <= 0:
        iImgWidth = max(1, int(iImgHeight / fAspect))
    # endif

    return iImgWidth, iImgHeight


# enddef
