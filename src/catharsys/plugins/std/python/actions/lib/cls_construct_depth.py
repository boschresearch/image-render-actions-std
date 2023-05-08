#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /do-tonemap.py
# Created Date: Thursday, October 22nd 2020, 4:26:28 pm
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

################################################################################
# Has to be called outside Blender
import os
import numpy as np
from pathlib import Path

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


from anybase import assertion
from anycam.obj import camera as camviewfactory

from catharsys.config.cls_project import CProjectConfig
import catharsys.util.config as cathcfg
import catharsys.util.file as cathfile
import catharsys.util.path as cathpath


################################################################################
class CConstructDepth:

    ################################################################################
    # Constants
    _iChX: int = 2
    _iChY: int = 1
    _iChZ: int = 0

    # Define output types and formats
    _dicTrgImgType: dict = {
        "radial-image": {"sFolder": "Radial", "sExt": ".exr"},
        "axis-z-image": {"sFolder": "AxisZ", "sExt": ".exr"},
        "preview": {"sFolder": "Preview", "sExt": ".png"},
        "debug": {"sFolder": "_debug", "sExt": ".png"},
    }

    ################################################################################
    # Member variables
    xPrjCfg: CProjectConfig = None
    dicConfig: dict = None
    dicData: dict = None

    sPathTrgMain: str = None
    pathSrcMain: Path = None
    dicPathTrgAct: dict = None
    dicActDtiToName: dict = None
    lActions: list = None
    iFrameFirst: int = None
    iFrameLast: int = None
    iFrameStep: int = None
    bDoProcess: bool = None
    bDoOverwrite: bool = None
    iDoProcess: int = None
    iDoOverwrite: int = None

    ################################################################################
    # Properties
    @property
    def dicTrgImgType(self) -> dict:
        return self._dicTrgImgType

    # enddef

    @property
    def iChX(self) -> int:
        return self._iChX

    # enddef

    @property
    def iChY(self) -> int:
        return self._iChY

    # enddef

    @property
    def iChZ(self) -> int:
        return self._iChZ

    # enddef

    ################################################################################
    def __init__(self):
        pass

    # enddef

    ################################################################################
    def Process(self, _xPrjCfg: CProjectConfig, _dicCfg: dict, **kwargs):
        assertion.FuncArgTypes()

        self.xPrjCfg = _xPrjCfg

        sWhere: str = "action configuration"
        self.dicConfig = cathcfg.GetDictValue(_dicCfg, "mConfig", dict, sWhere=sWhere)
        self.dicData = cathcfg.GetDictValue(self.dicConfig, "mData", dict, sWhere=sWhere)

        # Define expected type names
        sRenderTypeListDti = "blender/render/output-list:1"
        sConstructDepthDti = "blender/anytruth/construct/depth:1"

        lRndTypeList = cathcfg.GetDataBlocksOfType(self.dicData, sRenderTypeListDti)
        if len(lRndTypeList) == 0:
            raise Exception(
                "No render output configuration of type compatible to '{0}' given".format(sRenderTypeListDti)
            )
        # endif
        dicRndOutList = lRndTypeList[0]
        lRndOutTypes = cathcfg.GetDictValue(dicRndOutList, "lOutputs", list)
        if lRndOutTypes is None:
            raise Exception("No render output types defined")
        # endif

        # Look for 'anytruth/pos3d' render output type
        dicRndOut = None
        for dicOut in lRndOutTypes:
            dicRes = cathcfg.CheckConfigType(dicOut, "/catharsys/blender/render/output/anytruth/pos3d:1")
            if dicRes["bOK"] is True:
                dicRndOut = dicOut
                break
            # endif
        # endfor

        if dicRndOut is None:
            raise Exception("No render output type 'anytruth/pos3d' specified in configuration")
        # endif

        lCDT = cathcfg.GetDataBlocksOfType(self.dicData, sConstructDepthDti)
        if len(lCDT) == 0:
            raise Exception(
                "No label construction configuration of type compatible to '{0}' given".format(sConstructDepthDti)
            )
        # endif
        dicCDT = lCDT[0]

        # Get distance types
        lDistTypes = dicCDT.get("lDistTypes", ["radial"])

        cathcfg.StoreDictValuesInObject(
            self,
            _dicCfg,
            [
                ("sPathTrgMain", str),
                ("dicPathTrgAct", dict),
                ("dicActDtiToName", dict),
                ("lActions", list),
                ("iFrameFirst", int, 0),
                ("iFrameLast", int, 0),
                ("iFrameStep", int, 1),
                ("iDoProcess", int, 1),
                ("bDoProcess", bool, True),
                ("iDoOverwrite", int, 1),
                ("bDoOverwrite", bool, True),
            ],
            sWhere="action configuration",
        )

        self.bDoProcess = self.bDoProcess and (self.iDoProcess != 0)
        self.bDoOverwrite = self.bDoOverwrite and (self.iDoOverwrite != 0)

        # Get the name of the render action this action depends on
        sRenderActName = cathcfg.GetDictValue(
            self.dicActDtiToName, "action/std/blender/render/std:1", str, bIsDtiKey=True
        )
        # Get path where the render action, this action depends on, has generated its output.
        sPathRenderAct = cathcfg.GetDictValue(
            self.dicPathTrgAct,
            sRenderActName,
            str,
            sMsgNotFound="Render path not given for action {sKey}",
        )
        self.pathSrcMain = cathpath.MakeNormPath((sPathRenderAct, "AT_Pos3d_Raw"))

        print("Image source main path: {0}".format(self.pathSrcMain.as_posix()))
        print("Image target main path: {0}".format(self.sPathTrgMain))
        print("First rendered frame: {0}".format(self.iFrameFirst))
        print("Last rendered frame: {0}".format(self.iFrameLast))
        print("Frame step: {0}".format(self.iFrameStep))
        print("Do process: {0}".format(self.bDoProcess))
        print("Do overwrite: {0}".format(self.bDoOverwrite))

        ###################################################################################
        # Create output directories
        for sTrgImgType in self.dicTrgImgType:
            dicImgType = self.dicTrgImgType.get(sTrgImgType)
            sPathImgTrg = os.path.join(self.sPathTrgMain, dicImgType.get("sFolder"))

            if not os.path.exists(sPathImgTrg):
                cathpath.CreateDir(sPathImgTrg)
            # endif
            dicImgType["sPathImgTrg"] = sPathImgTrg
        # endfor

        # Loop over frames
        iTrgFrame = self.iFrameFirst - self.iFrameStep
        iTrgFrameIdx = -1
        # iTrgFrameCnt = int(math.floor((self.iFrameLast - self.iFrameFirst) / self.iFrameStep)) + 1

        while (iTrgFrame + self.iFrameStep) <= self.iFrameLast:

            # increment indices here to enable 'continue'
            # without need to increment at each continue command
            iTrgFrame += self.iFrameStep
            iTrgFrameIdx += 1

            print("")
            print("Start processing frame {0}...".format(iTrgFrame), flush=True)

            ###################################################################
            # Get pos3d raw info
            try:
                dicPos3d = cathcfg.Load(
                    (self.pathSrcMain, "Frame_{0:04d}.json".format(iTrgFrame)),
                    sDTI="/anytruth/render/pos3d/raw:1.1",
                )

            except Exception as xEx:
                print("Error reading pos3d config for frame {0}:\n{1}".format(iTrgFrame, str(xEx)))
                continue
            # endtry

            # lOffsetPos3d: list = dicPos3d.get("lOffsetPos3d")
            # if lOffsetPos3d is None:
            #     print("Error: element 'lOffsetPos3d' missing in pos3d data json")
            #     continue
            # # endif

            dicCamera = dicPos3d.get("mCamera")
            if dicCamera is None:
                print("Error: no camera information available for frame {0}".format(iTrgFrame))
                continue
            # endif

            ###################################################################
            sFrameName = "Frame_{0:04d}".format(iTrgFrame)
            sFileImgSrc = "{0}.exr".format(sFrameName)

            sFpImgSrc = os.path.normpath(os.path.join(self.pathSrcMain, sFileImgSrc))
            if not os.path.exists(sFpImgSrc):
                print("Frame '{0}' does not exist. Skipping...".format(sFileImgSrc))
                continue
            # endif

            # Check if target images already exist
            bTrgImgMissing = False

            for sTrgImgType in self.dicTrgImgType:
                dicImgType = self.dicTrgImgType.get(sTrgImgType)
                sPathImgTrg = dicImgType.get("sPathImgTrg")
                sFileImgTrg = "{0}{1}".format(sFrameName, dicImgType.get("sExt"))
                sFpImgTrg = os.path.join(sPathImgTrg, sFileImgTrg)
                dicImgType["sFileImgTrg"] = sFileImgTrg
                dicImgType["sFpImgTrg"] = sFpImgTrg

                if not os.path.exists(sFpImgTrg):
                    bTrgImgMissing = True
                elif self.bDoOverwrite:
                    print("Removing file due to overwrite flag: {0}".format(sFpImgTrg))
                    os.remove(sFpImgTrg)
                # endif
            # endfor

            if not bTrgImgMissing and not self.bDoOverwrite:
                print("Frame '{0}' already processed. Skipping...".format(sFrameName))
                continue
            # endif

            # Load source image
            # print("Loading image: {0}".format(sFpImgSrc))
            imgSrcImg = cv2.imread(
                sFpImgSrc,
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
            )
            if imgSrcImg is None:
                print("Error loading image: {0}".format(sFpImgSrc))
                continue
            # endif
            iSrcRows, iSrcCols, iSrcChnl = tSrcShape = imgSrcImg.shape

            # Create output arrays
            imgTrgRel = np.zeros((iSrcRows, iSrcCols, 3), dtype=np.float32)
            imgTrgRad = np.zeros((iSrcRows, iSrcCols), dtype=np.float32)
            imgTrgZ = np.zeros((iSrcRows, iSrcCols), dtype=np.float32)
            imgTrgPre = np.zeros((iSrcRows, iSrcCols), dtype=np.uint8)

            # typSrcImg = imgSrcImg.dtype

            ################################################################################
            # Evaluate distance to camera
            print("Evaluating depth from 3d data...", flush=True)

            imgMask = np.logical_or(np.abs(imgSrcImg[:, :, self.iChX]) > 0.0, np.abs(imgSrcImg[:, :, self.iChY]) > 0.0)
            imgMask = np.logical_or(imgMask, np.abs(imgSrcImg[:, :, self.iChZ]) > 0.0)
            imgMask = np.logical_and(imgMask, np.abs(imgSrcImg[:, :, self.iChX]) < 9e5)

            aRenderOffset = np.zeros(iSrcChnl)
            aOrig = np.zeros(iSrcChnl)
            aAxisZ = np.zeros(iSrcChnl)
            # aRenderOffset[[self.iChX, self.iChY, self.iChZ]] = np.array(lOffsetPos3d)
            aOrig[[self.iChX, self.iChY, self.iChZ]] = np.array(dicCamera.get("lOrigin"))
            aAxisZ[[self.iChX, self.iChY, self.iChZ]] = -np.array(dicCamera.get("lAxes")[2])

            # aRenderOffset = np.add(aRenderOffset, aOrig)
            aRenderOffset = aOrig
            imgTrgRel = np.subtract(imgSrcImg, aRenderOffset)
            imgTrgRad = np.sqrt(np.sum(np.square(imgTrgRel), axis=2))
            imgTrgZ = np.dot(imgTrgRel, aAxisZ)

            fMin = imgTrgRad.min(initial=1e9, where=imgMask)
            fMax = imgTrgRad.max(initial=-1e9, where=imgMask)
            imgTrgPre = (((imgTrgRad - fMin) / (fMax - fMin)) * 255.0).astype(np.uint8)
            # imgTrgPre[~imgMask] = 255
            imgTrgPre[~imgMask] = 0

            imgTrgRad[~imgMask] = 0.0
            imgTrgZ[~imgMask] = 0.0
            # imgTrgRad[~imgMask] = np.Inf
            # imgTrgZ[~imgMask] = np.Inf

            for sDistType in lDistTypes:
                if sDistType == "radial":
                    sFpImgTrg = self.dicTrgImgType.get("radial-image").get("sFpImgTrg")
                    cv2.imwrite(sFpImgTrg, imgTrgRad.astype(np.float32))
                elif sDistType == "axis-z":
                    sFpImgTrg = self.dicTrgImgType.get("axis-z-image").get("sFpImgTrg")
                    cv2.imwrite(sFpImgTrg, imgTrgZ.astype(np.float32))
                else:
                    print("Unsupported distance type '{0}'".format(sDistType))
                # endif
            # endfor

            sFpImgTrg = self.dicTrgImgType.get("preview").get("sFpImgTrg")
            cv2.imwrite(sFpImgTrg, imgTrgPre)

        # endwhile iTrgFrame

    # enddef


# endclass
