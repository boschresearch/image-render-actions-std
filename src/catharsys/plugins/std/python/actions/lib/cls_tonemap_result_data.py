#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \render_std_results.py
# Created Date: Tuesday, June 7th 2022, 2:54:10 pm
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

import os
from typing import Optional
from pathlib import Path

from anybase import assertion, config
from anybase.cls_any_error import CAnyError, CAnyError_Message
from catharsys.plugins.std.action_class.manifest.cls_cfg_manifest_job import (
    CConfigManifestJob,
)
from catharsys.plugins.std.resultdata import CImageResultData
from catharsys.plugins.std.blender.config.cls_compositor import CConfigCompositor
import catharsys.plugins.std.resultdata.util as resultutil


########################################################################################
class CTonemapResultData(CImageResultData):

    ####################################################################################
    def __init__(self, *, xJobCfg: CConfigManifestJob):

        # Member variable declaration
        self._dicImages: dict = None

        # Assert init function argument types
        assertion.FuncArgTypes()

        # Initialize parent class
        super().__init__(xJobCfg=xJobCfg)

    # enddef

    ##########################################################################
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
        iFrameFirst = config.GetDictValue(
            kwargs, "iFrameFirst", int, xDefault=0, sWhere=sWhere
        )
        iFrameLast = config.GetDictValue(
            kwargs, "iFrameLast", int, xDefault=0, sWhere=sWhere
        )
        iFrameStep = config.GetDictValue(
            kwargs, "iFrameStep", int, xDefault=1, sWhere=sWhere
        )
        bCheckImagesExist = config.GetDictValue(
            kwargs, "bCheckImagesExist", bool, xDefault=True, sWhere=sWhere
        )
        dicCfg = config.GetDictValue(
            kwargs, "dicCfg", dict, bOptional=True, sWhere=sWhere
        )

        self.ProcessImages(
            iFrameFirst=iFrameFirst,
            iFrameLast=iFrameLast,
            iFrameStep=iFrameStep,
            bCheckImagesExist=bCheckImagesExist,
            dicCfg=dicCfg,
        )

    # enddef

    ##########################################################################
    def ProcessImages(
        self,
        *,
        iFrameFirst: int = 0,
        iFrameLast: int = -1,
        iFrameStep: int = 1,
        bCheckImagesExist: bool = True,
        dicCfg: Optional[dict] = None,
    ):

        if iFrameFirst < 0:
            iStartIdx = 0
        else:
            iStartIdx = iFrameFirst
        # endif

        if iFrameLast < iStartIdx:
            iStopIdx = -1
        else:
            iStopIdx = iFrameLast
        # endif

        if iFrameStep < 0:
            iStepIdx = 1
        else:
            iStepIdx = iFrameStep
        # endif

        if dicCfg is None:
            lConfigs = self.xJobCfg.lConfigs
        else:
            lConfigs = [dicCfg]
        # endif

        self._dicImages = {}
        lTrgAction = ["/catharsys/action/std/blender/post-render/tonemap:1"]

        dicRdrTypeName = {
            "/catharsys/action/std/blender/post-render/tonemap:1": "tonemap"
        }

        for dicCfg in lConfigs:

            sPathTrgMain, iActIdx, sActionDti, sAction = self.xJobCfg._GetActionTrgPath(
                lTrgAction, dicCfg
            )
            if sPathTrgMain is None:
                continue
            # endif

            sRdrTypeName = config.GetDictValue(
                dicRdrTypeName, sActionDti, str, bIsDtiKey=True
            )
            pathTrgMain = Path(sPathTrgMain)

            iCfgIdx = dicCfg.get("iCfgIdx")
            iCfgCnt = dicCfg.get("iCfgCnt")

            dicRelPaths = self.xJobCfg._GetActionRelPaths(sAction, dicCfg)
            sRelPathTrial = dicRelPaths["sRelPathTrial"]
            sRelPathCfg = dicRelPaths["sRelPathCfg"]

            #########################################################
            # FIRST image dictionary level references:
            #       render configuration
            #########################################################
            dicImgCfg = self._dicImages.get(sRelPathCfg)
            if dicImgCfg is None:
                dicImgCfg = self._dicImages[sRelPathCfg] = {}
            # endif

            #########################################################
            # SECOND image dictionary level references:
            #       render sub type, e.g. label or depth
            #########################################################
            dicImgCfgRdr = dicImgCfg[sRdrTypeName] = {
                "iCfgIdx": iCfgIdx,
                "iCfgCnt": iCfgCnt,
                "sRelPathCfg": sRelPathCfg,
                "sRelPathTrial": sRelPathTrial,
                "mOutputType": {},
            }

            #########################################################
            # THIRD image dictionary level references:
            #       fixed name 'mOutputType',
            #       dictionary of output types as defined in compositor,
            #       or special output names like "AT_Label_Raw"
            #########################################################
            dicImgCfgRdrOut = dicImgCfgRdr.get("mOutputType")

            # Get expected image type name
            sType = dicRelPaths["lCfgIdFolders"][-1]

            #########################################################
            # FOURTH image dictionary level references:
            #       render output type, as defined in compositor
            #########################################################
            dicImgCfgRdrOutType = dicImgCfgRdrOut[sType] = {"mFrames": {}}
            #########################################################
            # FIFTH image dictionary level references:
            #       fixed name 'mFrames' that contains all frames
            #########################################################
            dicImgFrames = dicImgCfgRdrOutType.get("mFrames")

            resultutil.AddFramesToDictFromPath(
                pathFrames=pathTrgMain,
                dicImageFrames=dicImgFrames,
                sFileExt=".png",
                iFrameFirst=iStartIdx,
                iFrameLast=iStopIdx,
                iFrameStep=iStepIdx,
            )
        # endfor configurations

    # enddef


# endclass
