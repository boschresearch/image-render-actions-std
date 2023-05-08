#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \util_display_image_result_data.py
# Created Date: Wednesday, June 8th 2022, 5:16:17 pm
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

from typing import Optional
from anybase.ipy import CIPyRenderBase
from catharsys.plugins.std.util import imgproc


##########################################################################
def IPyRenderImageCfgRdrType(
    _xIpyRender: CIPyRenderBase,
    *,
    dicImgCfgRdr: dict,
    bAsVideo: bool = False,
    iFps: int = 10,
    iSubStart: int = 0,
    iSubCount: int = -1,
    iImgWidth: Optional[int] = None,
    iImgHeight: Optional[int] = None
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
                        sFpVideo = imgproc.CreateVideo(lFpImages=lSelFrames, iFps=iFps)

                        print("Video of sub-images written to: {0}".format(sFpVideo))
                        _xIpyRender.Video(
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
                _xIpyRender.Video(
                    sFpVideo=sFpVideo, iWidth=iImgWidth, iHeight=iImgHeight
                )
            # endif

        else:
            for iFrameIdx, sFpImage in enumerate(lFpImages):
                _xIpyRender.Text("[{0}] Frame {1}".format(sType, iFrameIdx))
                if sFpImage is not None:
                    _xIpyRender.Image(
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
                            _xIpyRender.Text(
                                "[{0}] Frame {1}[{2}/{3}]".format(
                                    sType,
                                    iFrameIdx,
                                    iSubStart + iSubIdx + 1,
                                    len(lSelFrames),
                                )
                            )
                            _xIpyRender.Image(
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
