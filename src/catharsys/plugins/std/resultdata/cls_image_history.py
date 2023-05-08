#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cls_image_result_data.py
# Created Date: Tuesday, June 7th 2022, 4:52:13 pm
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
from anybase import assertion, config
from anybase.cls_any_error import CAnyError, CAnyError_Message
from catharsys.api.cls_action_result_data import CActionResultData
from catharsys.plugins.std.action_class.manifest.cls_cfg_manifest_job import (
    CConfigManifestJob,
)

from catharsys.plugins.std.resultdata import CImageResultData

########################################################################################
class CImageResultDataHistory(CActionResultData):
    @property
    def xJobCfg(self) -> CConfigManifestJob:
        return self._xJobCfg

    @property
    def dicImages(self) -> dict:
        return self._dicImages

    ####################################################################################
    def __init__(self, *, xJobCfg: CConfigManifestJob):

        # Member variable declaration
        self._xJobCfg: CConfigManifestJob = None
        self._dicImages: dict = None

        # Assert init function argument types
        assertion.FuncArgTypes()

        # Initialize parent class
        super().__init__(
            sActionDti=xJobCfg.sActionDti,
            sDataDti="/catharsys/plugins/std/result-data/image-history:1.0",
        )

        # Member variable initialization
        self._xJobCfg = xJobCfg

    # enddef

    ####################################################################################
    def Process(self, **kwargs):

        lSupportedArgs = [
            "lRenderImageTypes",
            "iFrameFirst",
            "iFrameLast",
            "iFrameStep",
            "bCheckImagesExist",
            "dicCfg",
        ]

        for sArgKey in kwargs:
            if sArgKey not in lSupportedArgs:
                raise CAnyError_Message(
                    sMsg=(
                        f"Unsupported argument '{sArgKey}' for function Process()\n"
                        "Supported arguments are:"
                        + CAnyError.ListToString(lSupportedArgs)
                    )
                )
            # endif
        # endfor

        sWhere = "process function arguments"
        lRenderImageTypes = config.GetDictValue(
            kwargs, "lRenderImageTypes", list, xDefault=["*"], sWhere=sWhere
        )
        iFrameFirst = config.GetDictValue(
            kwargs, "iFrameFirst", int, xDefault=0, sWhere=sWhere
        )
        iFrameLast = config.GetDictValue(
            kwargs, "iFrameLast", int, xDefault=-1, sWhere=sWhere
        )
        iFrameStep = config.GetDictValue(
            kwargs, "iFrameStep", int, xDefault=1, sWhere=sWhere
        )
        bCheckImagesExist = config.GetDictValue(
            kwargs, "bCheckImagesExist", bool, xDefault=True, sWhere=sWhere
        )

        self.ProcessImages(
            lRenderImageTypes=lRenderImageTypes,
            iFrameFirst=iFrameFirst,
            iFrameLast=iFrameLast,
            iFrameStep=iFrameStep,
            bCheckImagesExist=bCheckImagesExist,
        )

    # enddef

    ##########################################################################
    def ProcessImages(
        self,
        *,
        lRenderImageTypes: list = ["*"],
        iFrameFirst: int = 0,
        iFrameLast: int = -1,
        iFrameStep: int = 1,
        bCheckImagesExist: bool = True,
    ):

        self._dicImages = {}
        for dicCfg in self.xJobCfg.lConfigs:

            dicActNameToDti = {}
            dicActDtiToName = dicCfg["dicActDtiToName"]
            for sActDti in dicActDtiToName:
                dicActNameToDti[dicActDtiToName[sActDti]] = sActDti
            # endfor

            lActionInfo = []
            dicActionImages = {}
            for sAction in dicCfg["lActions"]:

                sActionDti = dicActNameToDti[sAction]

                # Get result data object for action, expect type CImageResultData
                xResult: CImageResultData = self.xJobCfg.ResultData(
                    sActionDti=sActionDti
                )
                # If this is not result data based on CImageResultData, then ignore it
                if not isinstance(xResult, CImageResultData):
                    continue
                # endif

                dicActInfo = {"sAction": sAction, "sActionDti": sActionDti}
                dicActInfo.update(self.xJobCfg._GetActionRelPaths(sAction, dicCfg))
                lActionInfo.append(dicActInfo.copy())

                xResult.Process(
                    lRenderImageTypes=lRenderImageTypes,
                    iFrameFirst=iFrameFirst,
                    iFrameLast=iFrameLast,
                    iFrameStep=iFrameStep,
                    bCheckImagesExist=bCheckImagesExist,
                    dicCfg=dicCfg,
                )

                dicActionImages[sAction] = xResult.dicImages
            # endfor

            dicActInfo = lActionInfo[-1]
            sCfgId = "{}/{}".format(
                dicActInfo["sRelPathTrial"], dicActInfo["sRelPathCfg"]
            )

            self._dicImages[sCfgId] = {
                "iCfgIdx": dicCfg["iCfgIdx"],
                "iCfgCnt": dicCfg["iCfgCnt"],
                "lActionInfo": lActionInfo,
                "dicActionImages": dicActionImages,
            }
        # endfor

    # enddef


# endclass
