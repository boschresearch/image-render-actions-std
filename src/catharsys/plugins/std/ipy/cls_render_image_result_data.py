#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cls_display_image_result_data.py
# Created Date: Wednesday, June 8th 2022, 9:09:45 am
# Author: Christian Perwass (CR/AEC5)
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

from typing import Optional, Union, ForwardRef

from anybase import config
from anybase.ipy import CIPyRenderBase
from catharsys.plugins.std.resultdata import CImageResultData
from catharsys.plugins.std.util import imgproc

from .cls_image_render_base import CIPyImageRenderBase
from . import util_render_image_result_data as utilrender

TIPyRenderIRD = ForwardRef("CIPyRenderImageResultData")


########################################################################################
class CIPyRenderImageResultData(CIPyImageRenderBase):

    c_sText_ImageConfig: str = """
**Config {0}/{1}:** `{2}` [`{3}`]<br>
*Trial:* `{4}`
"""

    ####################################################################################
    def __init__(
        self,
        _xData: CImageResultData,
        *,
        xIpyRender: Union[CIPyImageRenderBase, TIPyRenderIRD, None] = None
    ):

        super().__init__(xIpyRender)

        self._xData = _xData

    # enddef

    ####################################################################################
    def Draw(self, **kwargs):
        sWhere = "Draw() function arguments"
        bAsVideo = config.GetDictValue(
            kwargs, "bAsVideo", bool, xDefault=False, sWhere=sWhere
        )
        iFps = config.GetDictValue(kwargs, "iFps", int, xDefault=10, sWhere=sWhere)
        iSubStart = config.GetDictValue(
            kwargs, "iSubStart", int, xDefault=0, sWhere=sWhere
        )
        iSubCount = config.GetDictValue(
            kwargs, "iSubCount", int, xDefault=-1, sWhere=sWhere
        )
        iImgWidth = config.GetDictValue(
            kwargs, "iImgWidth", int, xDefault=None, bOptional=True, sWhere=sWhere
        )
        iImgHeight = config.GetDictValue(
            kwargs, "iImgHeight", int, xDefault=None, bOptional=True, sWhere=sWhere
        )

        self.DrawImages(
            bAsVideo=bAsVideo,
            iFps=iFps,
            iSubStart=iSubStart,
            iSubCount=iSubCount,
            iImgWidth=iImgWidth,
            iImgHeight=iImgHeight,
        )

    # enddef

    ####################################################################################
    def DrawImages(
        self,
        *,
        bAsVideo=False,
        iFps=10,
        iSubStart=0,
        iSubCount=-1,
        iImgWidth=None,
        iImgHeight=None
    ):

        for sCfg in self._xData.dicImages:

            dicImgCfg = self._xData.dicImages.get(sCfg)

            for sRdrType in dicImgCfg:
                dicImgCfgRdr = dicImgCfg[sRdrType]
                iCfgIdx = dicImgCfgRdr.get("iCfgIdx")
                iCfgCnt = dicImgCfgRdr.get("iCfgCnt")
                sRelPathCfg = dicImgCfgRdr.get("sRelPathCfg")
                sRelPathTrial = dicImgCfgRdr.get("sRelPathTrial")

                self.Text(
                    self.c_sText_ImageConfig.format(
                        iCfgIdx + 1, iCfgCnt, sRelPathCfg, sRdrType, sRelPathTrial
                    )
                )

                utilrender.IPyRenderImageCfgRdrType(
                    self,
                    dicImgCfgRdr=dicImgCfgRdr,
                    bAsVideo=bAsVideo,
                    iFps=iFps,
                    iSubStart=iSubStart,
                    iSubCount=iSubCount,
                    iImgWidth=iImgWidth,
                    iImgHeight=iImgHeight,
                )
            # endfor render types
        # endfor configs

    # enddef

    ##########################################################################
    def _DispImgCfgRndType(
        self,
        *,
        dicImgCfgRdr,
        bAsVideo=False,
        iFps=10,
        iSubStart=0,
        iSubCount=-1,
        iImgWidth=None,
        iImgHeight=None
    ):

        dicImgCfgRdrOut = dicImgCfgRdr.get("mOutputType")
        for sType in dicImgCfgRdrOut:
            dicImgCfgRdrOutType = dicImgCfgRdrOut.get(sType)
            dicImgFrames = dicImgCfgRdrOutType.get("mFrames")

            lFpImages = []
            lFpSubImagesPerFrame = []
            for iFrameIdx in dicImgFrames:
                dicFrame = dicImgFrames.get(iFrameIdx)
                lFpImages.append(dicFrame.get("sFpImage"))
                lFpSubImagesPerFrame.append(dicFrame.get("lFpSubImages"))
            # endfor frames

            if bAsVideo:
                lFpValidImages = []
                lFpSubVideos = []
                for iImgIdx, sFpImage in enumerate(lFpImages):
                    if sFpImage is None:
                        lFrames = lFpSubImagesPerFrame[iImgIdx]
                        if iSubCount < 0:
                            lSelFrames = lFrames[iSubStart:]
                        elif iSubCount > 0:
                            iSubEnd = min(iSubStart + iSubCount, len(lFrames))
                            lSelFrames = lFrames[iSubStart:iSubEnd]
                        else:
                            lSelFrames = None
                        # endif

                        if lSelFrames is not None:
                            sFpVideo = imgproc.CreateVideo(
                                lFpImages=lSelFrames, iFps=iFps
                            )

                            print(
                                "Video of sub-images written to: {0}".format(sFpVideo)
                            )
                            self.Video(
                                sFpVideo=sFpVideo, iWidth=iImgWidth, iHeight=iImgHeight
                            )
                            lFpSubVideos.append(sFpVideo)
                        # endif
                    else:
                        lFpValidImages.append(sFpImage)
                    # endif
                # endfor

                if len(lFpValidImages) > 0:
                    sFpVideo = imgproc.CreateVideo(lFpImages=lFpValidImages, iFps=iFps)
                    print("Video written to: {0}".format(sFpVideo))
                    self.Video(sFpVideo=sFpVideo, iWidth=iImgWidth, iHeight=iImgHeight)
                # endif

            else:
                for iFrameIdx, sFpImage in enumerate(lFpImages):
                    self.Text("[{0}] Frame {1}".format(sType, iFrameIdx))
                    if sFpImage is not None:
                        self.Image(
                            sFpImage=sFpImage, iWidth=iImgWidth, iHeight=iImgHeight
                        )
                    # endif

                    lFpSubImages = lFpSubImagesPerFrame[iFrameIdx]
                    if isinstance(lFpSubImages, list) and len(lFpSubImages) > 0:
                        if iSubCount < 0:
                            lSelFrames = lFpSubImages[iSubStart:]
                        elif iSubCount > 0:
                            iSubEnd = min(iSubStart + iSubCount, len(lFpSubImages))
                            lSelFrames = lFpSubImages[iSubStart:iSubEnd]
                        else:
                            lSelFrames = None
                        # endif
                        if lSelFrames is not None:
                            for iSubIdx, sFpSubImage in enumerate(lSelFrames):
                                self.Text(
                                    "[{0}] Frame {1}[{2}/{3}]".format(
                                        sType,
                                        iFrameIdx,
                                        iSubStart + iSubIdx + 1,
                                        len(lSelFrames),
                                    )
                                )
                                self.Image(
                                    sFpImage=sFpSubImage,
                                    iWidth=iImgWidth,
                                    iHeight=iImgHeight,
                                )
                            # endfor
                        # endif
                    # endif
                # endfor frames
            # endif bAsVideo
        # endfor types

    # enddef


# endclass
