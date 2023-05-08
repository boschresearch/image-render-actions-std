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
from catharsys.plugins.std.resultdata import CImageResultDataHistory

from .cls_image_render_base import CIPyImageRenderBase
from . import util_render_image_result_data as utilrender

TIPyRenderIRDH = ForwardRef("CIPyRenderImageResultDataHistory")


########################################################################################
class CIPyRenderImageResultDataHistory(CIPyImageRenderBase):

    c_sText_ImageActCfg: str = """
**Action:** `{}` | **Config:** `[...]/{}` [`{}`] <br>
"""

    c_sText_BaseConfig: str = """
### Base Config {}/{}: `{}`<br>
**Trial:** `{}`
"""

    ####################################################################################
    def __init__(
        self,
        _xData: CImageResultDataHistory,
        *,
        xIPyRender: Union[CIPyImageRenderBase, TIPyRenderIRDH, None] = None
    ):

        super().__init__(xIPyRender)

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
        dicActDrawCfg = config.GetDictValue(
            kwargs, "dicActDrawCfg", dict, xDefault=None, bOptional=True, sWhere=sWhere
        )

        self.DrawImages(
            bAsVideo=bAsVideo,
            iFps=iFps,
            iSubStart=iSubStart,
            iSubCount=iSubCount,
            iImgWidth=iImgWidth,
            iImgHeight=iImgHeight,
            dicActDrawCfg=dicActDrawCfg,
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
        iImgHeight=None,
        dicActDrawCfg=None
    ):

        dicCfgImages = self._xData.dicImages
        if not isinstance(dicActDrawCfg, dict):
            dicActDrawCfg = {}
        # endif

        for sCfgId in dicCfgImages:
            dicCfgImg = dicCfgImages[sCfgId]
            iCfgIdx = dicCfgImg["iCfgIdx"]
            iCfgCnt = dicCfgImg["iCfgCnt"]
            lActionInfo = dicCfgImg["lActionInfo"]
            sBaseCfgId = lActionInfo[0]["sRelPathCfg"]
            sTrialId = lActionInfo[0]["sRelPathTrial"]
            self.Text(
                self.c_sText_BaseConfig.format(
                    iCfgIdx + 1, iCfgCnt, sBaseCfgId, sTrialId
                )
            )

            for dicActInfo in lActionInfo:
                sAction = dicActInfo["sAction"]
                sCfgId = dicActInfo["sRelPathCfg"]
                dicImages = dicCfgImg["dicActionImages"][sAction]
                dicImgCfg = dicImages.get(sCfgId)

                dicDrawCfg = dicActDrawCfg.get(sAction, {})

                for sRdrType in dicImgCfg:
                    dicImgCfgRdr = dicImgCfg[sRdrType]
                    iCfgIdx = dicImgCfgRdr.get("iCfgIdx")
                    iCfgCnt = dicImgCfgRdr.get("iCfgCnt")
                    sRelPathCfg = dicImgCfgRdr.get("sRelPathCfg")
                    # sRelPathTrial = dicImgCfgRnd.get("sRelPathTrial")

                    sSubPathCfg = sRelPathCfg[len(sBaseCfgId) :]
                    if len(sSubPathCfg) == 0:
                        sSubPathCfg = "."
                    elif sSubPathCfg.startswith("/"):
                        sSubPathCfg = sSubPathCfg[1:]
                    # endif

                    self.Text(
                        self.c_sText_ImageActCfg.format(sAction, sSubPathCfg, sRdrType)
                    )

                    utilrender.IPyRenderImageCfgRdrType(
                        self,
                        dicImgCfgRdr=dicImgCfgRdr,
                        bAsVideo=dicDrawCfg.get("bAsVideo", bAsVideo),
                        iFps=dicDrawCfg.get("iFps", iFps),
                        iSubStart=dicDrawCfg.get("iSubStart", iSubStart),
                        iSubCount=dicDrawCfg.get("iSubCount", iSubCount),
                        iImgWidth=dicDrawCfg.get("iImgWidth", iImgWidth),
                        iImgHeight=dicDrawCfg.get("iImgHeight", iImgHeight),
                    )
                # endfor render types
            # endfor
        # endfor

    # enddef


# endclass
