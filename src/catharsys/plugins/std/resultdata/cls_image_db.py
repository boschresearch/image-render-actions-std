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

import copy
import pickle
from timeit import default_timer as timer

from typing import Optional
from anybase import assertion, config
from catharsys.api import CResultData, CProject
from catharsys.plugins.std.action_class.manifest.cls_cfg_manifest_job import (
    CConfigManifestJob,
)

from .cls_image_history import CImageResultDataHistory

########################################################################################
class CImageResultDataDb(CResultData):
    @property
    def xProject(self) -> CProject:
        return self._xProject

    @property
    def dicImages(self) -> dict:
        return self._dicImages

    ####################################################################################
    def __init__(self, *, xProject: CProject):

        # Member variable declaration
        self._xProject: CProject = None
        self._dicImages: dict = None

        # Assert init function argument types
        assertion.FuncArgTypes()

        # Initialize parent class
        super().__init__(sDataDti="/catharsys/plugins/std/result-data/image-db:1.0")

        # Member variable initialization
        self._xProject = xProject

    # enddef

    ####################################################################################
    def Clear(self):
        self._dicImages = {}

    # enddef

    ####################################################################################
    def Save(self, sFilename):

        pathExport = (
            self.xProject.xConfig.pathOutput / self.xProject.xConfig.sLaunchFolderName
        )

        if not pathExport.exists():
            pathExport.mkdir(exist_ok=True)
        # endif

        pathFile = pathExport / sFilename

        print("Writing database to: {}".format(pathFile.as_posix()))

        with open(pathFile.as_posix(), "wb") as xFile:
            pickle.dump(self.dicImages, xFile)
        # endwith

    # enddef

    ####################################################################################
    def Load(self, sFilename, *, bDoRaise=True):

        pathExport = (
            self.xProject.xConfig.pathOutput / self.xProject.xConfig.sLaunchFolderName
        )

        pathFile = pathExport / sFilename
        if not pathFile.exists():
            if bDoRaise is True:
                raise RuntimeError(
                    "File does not exist: {}".format(pathFile.as_posix())
                )
            else:
                return False
            # endif
        # endif

        with open(pathFile.as_posix(), "rb") as xFile:
            self.dicImages = pickle.load(xFile)
        # endwith

        return True

    # enddef

    ####################################################################################
    def _ProvideDicEl(self, _dicX, _sEl, _xDefault):
        xEl = _dicX.get(_sEl)
        if xEl is None:
            xEl = _dicX[_sEl] = _xDefault
        # endif

        return xEl

    # enddef

    ####################################################################################
    def Process(self, **kwargs):

        sWhere = "process function arguments"
        lJobs = config.GetDictValue(kwargs, "lJobs", list, sWhere=sWhere)

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

        self.ProcessJobs(
            lJobs=lJobs,
            lRenderImageTypes=lRenderImageTypes,
            iFrameFirst=iFrameFirst,
            iFrameLast=iFrameLast,
            iFrameStep=iFrameStep,
            bCheckImagesExist=bCheckImagesExist,
        )

    # enddef

    ##########################################################################
    def ProcessJobs(
        self,
        *,
        lJobs: list[CConfigManifestJob],
        lRenderImageTypes: list = ["*"],
        iFrameFirst: int = 0,
        iFrameLast: int = -1,
        iFrameStep: int = 1,
        bCheckImagesExist: bool = True
    ):

        self._dicImages = {}

        # Process all jobs to obtain the image result data history
        lJobHistData: list[CImageResultDataHistory] = []
        for xJobCfg in lJobs:
            print(
                "Get job image result data history for action: {}".format(
                    xJobCfg.sAction
                )
            )
            dTimeStart = timer()
            xData = CImageResultDataHistory(xJobCfg=xJobCfg)
            xData.ProcessImages(
                lRenderImageTypes=lRenderImageTypes,
                iFrameFirst=iFrameFirst,
                iFrameLast=iFrameLast,
                iFrameStep=iFrameStep,
                bCheckImagesExist=bCheckImagesExist,
            )
            lJobHistData.append(xData)
            dTimeEnd = timer()
            print("...done in {:5.2f}s".format(dTimeEnd - dTimeStart))
        # endfor

        for xData in lJobHistData:
            dicActHistImg = xData.dicImages

            for sCfgImg in dicActHistImg:
                dicCfgImg = dicActHistImg[sCfgImg]

                lActInfo = dicCfgImg["lActionInfo"]
                sTrialId = lActInfo[0]["sRelPathTrial"]
                sBaseCfgId = lActInfo[0]["sRelPathCfg"]
                dicActImg = dicCfgImg["dicActionImages"]

                dicTrgTrial = self._ProvideDicEl(self._dicImages, sTrialId, {})
                dicTrgBaseCfg = self._ProvideDicEl(dicTrgTrial, sBaseCfgId, {})

                for dicActInfo in lActInfo:
                    sAction = dicActInfo["sAction"]
                    sRelPathCfg = dicActInfo["sRelPathCfg"]

                    dicAct = dicActImg[sAction]
                    dicRndTypes = dicAct[sRelPathCfg]

                    for sRdrType in dicRndTypes:
                        dicRdrType = dicRndTypes[sRdrType]
                        dicTrgRdrType = self._ProvideDicEl(dicTrgBaseCfg, sRdrType, {})

                        sRelPathCfg = dicRdrType["sRelPathCfg"]
                        sSubPathCfg = sRelPathCfg[len(sBaseCfgId) :]
                        if len(sSubPathCfg) == 0:
                            sSubPathCfg = "."
                        elif sSubPathCfg.startswith("/"):
                            sSubPathCfg = sSubPathCfg[1:]
                        # endif
                        dicTrgRdrType["sSubPathCfg"] = sSubPathCfg

                        dicRndOutTypes = dicRdrType["mOutputType"]
                        dicTrgRdrOutTypes = self._ProvideDicEl(
                            dicTrgRdrType, "mOutputType", {}
                        )

                        for sRndOutType in dicRndOutTypes:
                            dicRndOutType = dicRndOutTypes[sRndOutType]
                            dicTrgRdrOutType = self._ProvideDicEl(
                                dicTrgRdrOutTypes, sRndOutType, {}
                            )
                            dicFrames = dicRndOutType["mFrames"]
                            dicTrgFrames = self._ProvideDicEl(
                                dicTrgRdrOutType, "mFrames", {}
                            )
                            for sFrame in dicFrames:
                                if sFrame in dicTrgFrames:
                                    continue
                                # endif
                                dicTrgFrames[sFrame] = copy.deepcopy(dicFrames[sFrame])
                            # endfor frames
                        # endfor render output types
                    # endfor render types
                # endfor actions
            # endfor configs
        # endfor job history data

    # enddef


# endclass
