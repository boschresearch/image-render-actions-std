#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /do-construct-rs.py
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
import math
import numpy as np
from pathlib import Path

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import enum
from timeit import default_timer as timer

from anybase import assertion
from anybase import time as anytime
from anybase.cls_any_error import CAnyError_Message
from catharsys.config.cls_config_list import CConfigList
from catharsys.config.cls_project import CProjectConfig
from catharsys.plugins.std.blender.config.cls_compositor import CConfigCompositor
from catharsys.plugins.std.blender.actions.lib.cls_rsexp import CRsExp
import catharsys.util as cathutil
import catharsys.util.config as cathcfg
import catharsys.util.file as cathfile
import catharsys.util.path as cathpath


################################################################################
class ERndOutType(enum.Enum):
    NONE = enum.auto()
    IMAGE = enum.auto()
    POS3D = enum.auto()


# endclass


################################################################################
class CConstructRS:
    xPrjCfg: CProjectConfig = None

    dicConfig: dict = None
    sPathTrgMain: str = None
    dicPathTrgAct: dict = None
    dicActDtiToName: dict = None
    iFrameFirst: int = None
    iFrameLast: int = None
    iFrameStep: int = None
    iDoProcess: int = None
    bDoProcess: bool = None
    bDoEval: bool = None
    dicData: dict = None

    dicRsExp: dict = None
    fSrcFps: float = None
    fSrcFrameTime: float = None
    iLineCount: int = None
    fScnFps: float = None
    iSrcReadOutsPerRender: int = None
    dicSrcExp: dict = None
    fSrcExpPerLine: float = None

    iChX: int = 2
    iChY: int = 1
    iChZ: int = 0

    ################################################################################
    def __init__(self):
        pass

    # enddef

    ################################################################################
    def Process(self, _xPrjCfg: CProjectConfig, _dicCfg: dict, **kwargs):
        assertion.FuncArgTypes()

        self.xPrjCfg = _xPrjCfg

        cathcfg.StoreDictValuesInObject(
            self,
            _dicCfg,
            [
                ("dicConfig", "mConfig", dict),
                ("sPathTrgMain", str),
                ("dicPathTrgAct", dict),
                ("dicActDtiToName", dict),
                ("iFrameFirst", int, 0),
                ("iFrameLast", int, 0),
                ("iFrameStep", int, 1),
                ("iDoProcess", int, 1),
                ("bDoProcess", bool, True),
            ],
            sWhere="action configuration",
        )

        self.bDoEval = self.bDoProcess and (self.iDoProcess != 0)
        self.dicData = cathcfg.GetDictValue(self.dicConfig, "mData", dict, sWhere="action configuration")

        # Define expected type names
        sDtiConstruct = "blender/construct/rs/type/*:1"
        sDtiImgType = "blender/construct/rs/input-id:1"
        sDtiRenderOutput = "blender/render/output/*:1"
        sDtiRenderOutputList = "blender/render/output-list:1"

        lRndTypeList = cathcfg.GetDataBlocksOfType(self.dicData, sDtiRenderOutputList)
        if len(lRndTypeList) == 0:
            raise Exception(
                "No render output configuration of type compatible to '{0}' given".format(sDtiRenderOutputList)
            )
        # endif
        dicRndOutList = lRndTypeList[0]
        lRndOutTypes = cathcfg.GetDictValue(dicRndOutList, "lOutputs", list)
        if lRndOutTypes is None:
            raise Exception("No render output types defined")
        # endif

        if len(lRndOutTypes) > 1:
            raise Exception("Currently only a single output type is allowed for constructing rolling shutter")
        # endif

        if len(lRndOutTypes) == 0:
            raise Exception("No render output types defined")
        # endif

        dicRndOut = lRndOutTypes[0]
        dicRes = cathcfg.CheckConfigType(dicRndOut, sDtiRenderOutput)
        if dicRes["bOK"] is False:
            raise RuntimeError("Invalid render output type")
        # endif

        lSpecificRenderType: list[str] = dicRes["lCfgType"][4:]

        eSrcRndOutType = ERndOutType.NONE
        if lSpecificRenderType[0] == "image":
            eSrcRndOutType = ERndOutType.IMAGE
        elif lSpecificRenderType[0] == "anytruth" and lSpecificRenderType[1] == "pos3d":
            eSrcRndOutType = ERndOutType.POS3D
        else:
            sSpecRndOutType = "/".join(lSpecificRenderType)
            raise RuntimeError(
                f"Unsupported source render output type '{sSpecRndOutType}' for rolling shutter construction."
            )
        # endif

        lCtr: list[dict] = cathcfg.GetDataBlocksOfType(self.dicData, sDtiConstruct)
        if len(lCtr) == 0:
            raise Exception("No construct configuration of type compatible to '{0}' given.".format(sDtiConstruct))
        # endif
        dicConstruct: dict = lCtr[0]
        dicConstructDti: dict = cathcfg.SplitDti(cathcfg.GetDictValue(dicConstruct, "sDTI", str))
        lConstructType: list[str] = dicConstructDti["lType"][5:]
        iReadOutsPerRender: int = cathcfg.GetDictValue(dicConstruct, "iReadOutsPerRender", int, xDefault=1)

        # Get the name of the render action this action depends on
        sRenderActName = cathcfg.GetDictValue(
            self.dicActDtiToName, "action/std/blender/render/rs:1", str, bIsDtiKey=True
        )
        # Get path where the render action, this action depends on, has generated its output.
        sPathRenderAct = cathcfg.GetDictValue(
            self.dicPathTrgAct,
            sRenderActName,
            str,
            sMsgNotFound="Rolling shutter render path not give for action {sKey}",
        )

        # Load RS config data stored by RS render script
        dicRsCfg = cathcfg.Load((sPathRenderAct, "RsCfg.json"), sDTI="rs-config:1.1")

        # Load AnyCam config stored by RS render script
        sFpAnyCamCfg = os.path.join(sPathRenderAct, "AnyCam.json")
        bFound = os.path.exists(sFpAnyCamCfg)
        # print("AnyCam config file: {0} -> {1}".format(sFpAnyCamCfg, "found" if bFound else "not found"))
        if not bFound:
            raise Exception("AnyCam config file not found at: {0}".format(sFpAnyCamCfg))
        # endif
        dicAnyCam = cathfile.LoadJson(sFpAnyCamCfg)

        ###############################################################################
        # Image RS setup
        if eSrcRndOutType == ERndOutType.IMAGE:
            dicComp = cathcfg.GetDictValue(dicRndOut, "mCompositor", dict)
            cathcfg.AssertConfigType(dicComp, "/catharsys/blender/compositor:1")
            xComp = CConfigCompositor(xPrjCfg=self.xPrjCfg, dicData=dicComp)

            lImgFolder: list[str] = cathcfg.GetDataBlocksOfType(self.dicData, sDtiImgType)
            if len(lImgFolder) == 0:
                raise Exception("No image type configuration of type compatible to '{0}' given.".format(sDtiImgType))
            # endif
            sImageFolder = lImgFolder[0]

            # Get compositor file format for given folder name
            dicCompFo = xComp.GetOutputsByFolderName()
            lImageFolderFo = cathcfg.GetDictValue(
                dicCompFo,
                sImageFolder,
                list,
                xDefault=[],
                sWhere="compositor file format by folder",
            )
            if len(lImageFolderFo) == 0:
                raise Exception(
                    "Compositor configuration does not contain an output of type '{0}'.".format(sImageFolder)
                )
            elif len(lImageFolderFo) > 1:
                raise Exception(
                    "Compositor configuration contains more than one output to the folder '{0}'.".format(sImageFolder)
                )
            # endif
            dicImageFolderFo = lImageFolderFo[0]

        # #########################################
        # Pos3d RS setup
        elif eSrcRndOutType == ERndOutType.POS3D:
            dicImageFolderFo = {
                "sContentType": "pos3d",
                "sFolder": "AT_Pos3d_Raw",
                "sFileExt": ".exr",
            }

        # #########################################
        # Error case
        else:
            raise RuntimeError(f"Unsupported source render output type '{(str(eSrcRndOutType))}")
        # endif

        ###################################################################################
        # Prepare image capture data for processing

        dicExposure = cathcfg.GetDictValue(dicConstruct, "mExp", dict, sWhere="construction configuration")
        if dicExposure is None:
            raise Exception("No exposure definitions given in image construction configuration")
        # endif

        sConstructId = cathcfg.GetDictValue(dicConstruct, "sId", str, sWhere="construction configuration")

        # Load AnyCam config
        iSenResX = cathcfg.GetDictValue(dicAnyCam, "iSenResX", int, sWhere="AnyCam dictionary")
        iSenResY = cathcfg.GetDictValue(dicAnyCam, "iSenResY", int, sWhere="AnyCam dictionary")
        lTrgImgSize = (iSenResY, iSenResX, 3)

        # Load RS render config
        sWhere = "rolling shutter configuration"
        iSrcBorderTop = cathcfg.GetDictValue(dicRsCfg, "iBorderTop", int, sWhere=sWhere)
        self.dicRsExp = cathcfg.GetDictValue(dicRsCfg, "mRsExp", dict, sWhere=sWhere)

        cathcfg.StoreDictValuesInObject(
            self,
            self.dicRsExp,
            [
                ("fSrcFps", "dTrgFps", float),
                ("fSrcFrameTime", "dTrgFrameTime", float),
                ("iLineCount", int),
                ("fScnFps", "dScnFps", float),
                ("iSrcReadOutsPerRender", "iReadOutsPerRender", int, 1),
                ("dicSrcExp", "mTrgExp", dict),
            ],
            sWhere=sWhere,
        )

        self.fSrcExpPerLine = cathcfg.GetDictValue(self.dicSrcExp, "dExpPerLine", float, sWhere=sWhere)

        iTrgPerSrcReadOutSteps = self.iSrcReadOutsPerRender / iReadOutsPerRender
        iTrgReadOutStepIdx = 0

        xSrcRsExp = CRsExp(
            fFPS=self.fSrcFps,
            fFrameTime=self.fSrcFrameTime,
            iLineCount=self.iLineCount,
            fScnFps=self.fScnFps,
            iReadOutsPerRender=self.iSrcReadOutsPerRender,
            dicExp=self.dicSrcExp,
        )
        print("====================================================")
        print("Rolling shutter source configuration\n")
        xSrcRsExp.PrintData()

        lTrgRsExp: list[CRsExp] = []
        lConstructCfg: list[dict] = []

        # ################################################################################################
        # Construction type "single"
        # All lines are combined to achieve the exposure time per line
        if lConstructType[0] == "single":
            dTrgExpPerLine = cathcfg.GetDictValue(
                dicExposure,
                "dExpPerLine",
                float,
                xDefault=0.0,
                sWhere="exposure configuration",
            )
            if dTrgExpPerLine > self.fSrcExpPerLine:
                raise Exception(
                    "Given exposure time '{0}' is larger than rendered exposure time '{1}'.".format(
                        dTrgExpPerLine, self.fSrcExpPerLine
                    )
                )
            elif dTrgExpPerLine <= 0.0:
                raise Exception("Invalid exposure time '{0}'.".format(dTrgExpPerLine))
            # endif

            xTrgRsExp = CRsExp(
                fFPS=self.fSrcFps,
                fFrameTime=self.fSrcFrameTime,
                iLineCount=self.iLineCount,
                fScnFps=self.fScnFps * self.iSrcReadOutsPerRender,
                iReadOutsPerRender=iReadOutsPerRender,
                dicExp=dicExposure,
            )

            lTrgRsExp.append(xTrgRsExp)
            xTrgRsExp.PrintData()

            iReadOutsPerExp = xTrgRsExp.GetReadOutsPerExp()
            dEffExpPerLine = xTrgRsExp.GetEffExpPerLine()

            # Store the collected construction parameters in JSON file with constructed images
            lConstructCfg.append(
                {
                    "sId": sConstructId,
                    "sType": "RS",
                    "mExp": dicExposure.copy(),
                    "dTrgFrameTime": self.fSrcFrameTime,
                    "iLineCount": self.iLineCount,
                    "iReadOutsPerRender": iReadOutsPerRender,
                    "dScnFps": self.fScnFps,
                    "iSrcBorderTop": iSrcBorderTop,
                    "dMaxTrgExpPerLine": self.fSrcExpPerLine,
                    "dTrgExpPerLine": dTrgExpPerLine,
                    "iReadOutsPerExp": iReadOutsPerExp,
                    "dEffExpPerLine": dEffExpPerLine,
                    "iSenResX": iSenResX,
                    "iSenResY": iSenResY,
                    "mAnyCam": dicAnyCam.copy()
                    # "": ,
                }
            )
            sDtiConstruct = "construct-proc/rs/type/single:1"

        # ################################################################################################
        # Construction type "consecutive"
        # Lines are combined to simulate a set of exposures per frame.
        # This can be used to simulate HDR exposures where a set of consecutive exposures,
        # of increasing time are created. The construction will output one image per exposure time,
        # which later can be combined in an HDR image by an appropriate algorithm.
        elif lConstructType[0] == "consecutive":
            lTrgExpPerLine = cathcfg.GetDictValue(dicExposure, "lExpPerLine", list, sWhere="exposure configuration")

            dExpOffset = cathcfg.GetDictValue(
                dicExposure,
                "dExpOffset",
                float,
                xDefault=0.0,
                sWhere="exposure configuration",
            )
            lReadOutLinePattern = cathcfg.GetDictValue(
                dicExposure,
                "lReadOutLinePattern",
                list,
                xDefault=[0],
                sWhere="exposure configuration",
            )

            dTotalTrgExpPerLine = sum(lTrgExpPerLine)
            if dTotalTrgExpPerLine > self.fSrcExpPerLine:
                raise Exception(
                    "Given total exposure time '{0}' is larger than rendered exposure time '{1}'.".format(
                        dTotalTrgExpPerLine, self.fSrcExpPerLine
                    )
                )
            elif dTotalTrgExpPerLine <= 0.0:
                raise Exception("Invalid exposure time '{0}'.".format(dTotalTrgExpPerLine))
            # endif

            dicExp = {
                "lReadOutLinePattern": lReadOutLinePattern,
                "dExpOffset": 0,
                "dExpPerLine": 0,
            }

            dTrgExpPerLineSum = 0.0
            for dTrgExpPerLine in lTrgExpPerLine:
                dicExp["dExpOffset"] = dExpOffset - dTrgExpPerLineSum
                dicExp["dExpPerLine"] = dTrgExpPerLine
                dTrgExpPerLineSum += dTrgExpPerLine

                xTrgRsExp = CRsExp(
                    fFPS=self.fSrcFps,
                    fFrameTime=self.fSrcFrameTime,
                    iLineCount=self.iLineCount,
                    fScnFps=self.fScnFps * self.iSrcReadOutsPerRender,
                    iReadOutsPerRender=iReadOutsPerRender,
                    dicExp=dicExp,
                )
                lTrgRsExp.append(xTrgRsExp)

                print("====================================================")
                print("Rolling shutter construct for line exposure: {}\n".format(dTrgExpPerLine))
                xTrgRsExp.PrintData()

                iReadOutsPerExp = xTrgRsExp.GetReadOutsPerExp()
                dEffExpPerLine = xTrgRsExp.GetEffExpPerLine()

                # Store the collected construction parameters in JSON file with constructed images
                lConstructCfg.append(
                    {
                        "sId": sConstructId,
                        "sType": "RS",
                        "mExp": dicExp.copy(),
                        "dTrgFrameTime": self.fSrcFrameTime,
                        "iLineCount": self.iLineCount,
                        "iReadOutsPerRender": iReadOutsPerRender,
                        "dScnFps": self.fScnFps,
                        "iSrcBorderTop": iSrcBorderTop,
                        "dMaxTrgExpPerLine": self.fSrcExpPerLine,
                        "lTrgExpPerLine": lTrgExpPerLine,
                        "iReadOutsPerExp": iReadOutsPerExp,
                        "dEffExpPerLine": dEffExpPerLine,
                        "iSenResX": iSenResX,
                        "iSenResY": iSenResY,
                        "mAnyCam": dicAnyCam.copy()
                        # "": ,
                    }
                )
            # endfor

            sDtiConstruct = "construct-proc/rs/type/consecutive:1"

        else:
            raise RuntimeError("Unsupported rolling shutter construction type '{}'".format(lConstructType[0]))
        # endif

        # Create dirs for all render output of given id
        if eSrcRndOutType == ERndOutType.IMAGE:
            if len(lTrgRsExp) == 1:
                lPathTrgMain = [self.sPathTrgMain]
                cathpath.CreateDir(self.sPathTrgMain)
                cathcfg.Save(
                    (self.sPathTrgMain, "ConstructCfg.json"),
                    lConstructCfg[0],
                    sDTI=sDtiConstruct,
                )
            else:
                lPathTrgMain = []
                for iRsIdx in range(len(lTrgRsExp)):
                    sPath = os.path.join(self.sPathTrgMain, str(iRsIdx))
                    lPathTrgMain.append(sPath)
                    cathpath.CreateDir(sPath)
                    cathcfg.Save(
                        (sPath, "ConstructCfg.json"),
                        lConstructCfg[iRsIdx],
                        sDTI=sDtiConstruct,
                    )
                # endfor
            # endif

        elif eSrcRndOutType == ERndOutType.POS3D:
            pathTrgMain = Path(self.sPathTrgMain)
            pathTrgMain = pathTrgMain.parent / "AT_Depth"

            if len(lTrgRsExp) == 1:
                lPathTrgMain = [pathTrgMain.as_posix()]
                pathTrgMain.mkdir(exist_ok=True, parents=True)
                cathcfg.Save(
                    (pathTrgMain.as_posix(), "ConstructCfg.json"),
                    lConstructCfg[0],
                    sDTI=sDtiConstruct,
                )

            else:
                lPathTrgMain = []
                for iRsIdx in range(len(lTrgRsExp)):
                    pathTrgRs = pathTrgMain / str(iRsIdx)
                    pathTrgRs.mkdir(exist_ok=True, parents=True)
                    lPathTrgMain.append(pathTrgRs.as_posix())
                    cathcfg.Save(
                        (pathTrgRs.as_posix(), "ConstructCfg.json"),
                        lConstructCfg[0],
                        sDTI=sDtiConstruct,
                    )
                # endfor
            # endif
        else:
            raise RuntimeError("Unsupported source render output type")
        # endif

        # Loop over frames
        iTrgFrame = self.iFrameFirst
        iTrgFrameIdx = 0
        iTrgFrameCnt = int(math.floor((self.iFrameLast - self.iFrameFirst) / self.iFrameStep)) + 1

        # Number of exposures to construct
        iRsCnt = len(lTrgRsExp)
        for xTrgRsExp in lTrgRsExp:
            xTrgRsExp.SetTrgFrame(iTrgFrame)
        # endfor
        xSrcRsExp.SetTrgFrame(iTrgFrame)
        iRoLoopCnt = xSrcRsExp.GetReadOutLoopCount() * iTrgPerSrcReadOutSteps
        iTotalRoLoopCnt = iTrgFrameCnt * iRoLoopCnt

        while True:
            if iTrgFrame > self.iFrameLast:
                break
            # endif

            print("")
            print("=============================================")
            print("Start processing frame {0}...".format(iTrgFrame))
            print("")

            # Need to reset this for each new frame, so that the
            # xSrcRsExp steps in sync with xTrgRsExp.
            iTrgReadOutStepIdx = 0

            xSrcRsExp.SetTrgFrame(iTrgFrame)
            for xTrgRsExp in lTrgRsExp:
                xTrgRsExp.SetTrgFrame(iTrgFrame)
            # endfor

            # Create the render path for this frame
            sFrameName = "Frame_{0:04d}".format(iTrgFrame)

            sPathRenderFrame = os.path.join(sPathRenderAct, sFrameName)

            # Content type of rendered file
            sContentType = cathcfg.GetDictValue(dicImageFolderFo, "sContentType", str)
            sImgSrcFolder = cathcfg.GetDictValue(dicImageFolderFo, "sFolder", str)
            sImgSrcExt = cathcfg.GetDictValue(dicImageFolderFo, "sFileExt", str)

            # Create empty result image for frame
            lImgTrg = []
            for iRsIdx in range(iRsCnt):
                if sContentType == "image":
                    lImgTrg.append(np.zeros(lTrgImgSize, np.float32))
                elif sContentType == "depth" or sContentType == "pos3d":
                    lImgTrg.append(np.zeros(lTrgImgSize, np.float64))
                else:
                    raise Exception("Output content type '{0}' not supported.".format(sContentType))
                # endif
            # endfor

            iRoLoopIdx = 0
            sTimeDelta = "n/a"
            sTimeLeft = "n/a"
            dTimeStart = timer()

            xSrcRsExp.StartReadOutLoop()

            # Only start RS loop if at least one exposure has a valid loop
            lValidRsLoop = []
            for xTrgRsExp in lTrgRsExp:
                bStart = xTrgRsExp.StartReadOutLoop()
                lValidRsLoop.append(bStart)
            # endfor

            if any(lValidRsLoop) is True:
                bLoadImage = True
                imgSrc = None

                while True:
                    iTotalRoLoopIdx = iTrgFrameIdx * iRoLoopCnt + iRoLoopIdx
                    dRoLoopPart = 100.0 * (iRoLoopIdx / iRoLoopCnt)
                    dTotalRoLoopPart = 100.0 * (iTotalRoLoopIdx / iTotalRoLoopCnt)

                    print("=============================================================")
                    print("Source Ro Idx: {}".format(xSrcRsExp.iReadOutIdx))

                    iExpStartScnFrame = xSrcRsExp.GetExpStartSceneFrame()
                    # print("iExpStartScnFrame: {}".format(iExpStartScnFrame))

                    # Check exposure image
                    sRelPathExpImg = "{0}/Exp_{1:07d}{2}".format(sImgSrcFolder, iExpStartScnFrame, sImgSrcExt)
                    sFpExpImg = os.path.normpath(os.path.join(sPathRenderFrame, sRelPathExpImg))
                    bLoadImage = True
                    if not os.path.exists(sFpExpImg):
                        # Test whether an image has been previously loaded
                        if imgSrc is None:
                            raise Exception(
                                "Image file for frame {0}, exposure {1} does not exist. ({2})".format(
                                    iTrgFrame, iExpStartScnFrame, sFpExpImg
                                )
                            )
                        # endif

                        # Reuse previously loaded image, assuming it contains usable data
                        bLoadImage = False
                    # endif

                    if iRoLoopIdx > 0:
                        dTimeNow = timer()
                        dTimeDelta = dTimeNow - dTimeStart
                        sTimeDelta = anytime.SecondsToHmsStr(dTimeDelta)
                        dTimeLeft = (iTotalRoLoopCnt / iTotalRoLoopIdx - 1.0) * dTimeDelta
                        sTimeLeft = anytime.SecondsToHmsStr(dTimeLeft)
                    # endif

                    if bLoadImage is True:
                        print(
                            f"Time: {sTimeDelta} + {sTimeLeft} "
                            f"| Frame {iTrgFrame:2d} ({(iTrgFrameIdx + 1):2d}/{iTrgFrameCnt:2d}) "
                            f"| {dTotalRoLoopPart:5.1f}% "
                            f"| {dRoLoopPart:5.1f}% "
                            f"| Loading and processing exposure image '{sRelPathExpImg}'...",
                            flush=True,
                        )
                    else:
                        print(
                            f"Time: {sTimeDelta} + {sTimeLeft} "
                            f"| Frame {iTrgFrame:2d} ({(iTrgFrameIdx + 1):2d}/{iTrgFrameCnt:2d}) "
                            f"| {dTotalRoLoopPart:5.1f}% "
                            f"| {dRoLoopPart:5.1f}% "
                            f"| Reusing previous image...",
                            flush=True,
                        )
                    # endif

                    if self.bDoEval:
                        if bLoadImage is True:
                            imgSrc: np.ndarray = cv2.imread(
                                sFpExpImg,
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                            )

                            if eSrcRndOutType == ERndOutType.POS3D:
                                sRelPathExpImg = "{0}/Exp_{1:07d}.json".format(sImgSrcFolder, iExpStartScnFrame)
                                sFpExpData = os.path.normpath(os.path.join(sPathRenderFrame, sRelPathExpImg))

                                ###################################################################
                                # Get pos3d raw info
                                try:
                                    print(f"Loading camera data from: {sFpExpData}")
                                    dicPos3d = cathcfg.Load(
                                        sFpExpData,
                                        sDTI="/anytruth/render/pos3d/raw:1.1",
                                    )

                                except Exception as xEx:
                                    raise CAnyError_Message(
                                        sMsg=f"Error reading pos3d data for exposure frame {iExpStartScnFrame} "
                                        f"of target frame {iTrgFrame}:\n{(str(xEx))}",
                                        xChildEx=xEx,
                                    )
                                # endtry

                                dicCamera: dict = dicPos3d.get("mCamera")
                                if dicCamera is None:
                                    raise RuntimeError(
                                        "Error: no camera information available for frame {0}".format(iExpStartScnFrame)
                                    )
                                # endif

                                ###################################################################

                            # endif

                        # endif

                        for iRsIdx, xTrgRsExp in enumerate(lTrgRsExp):
                            if lValidRsLoop[iRsIdx] is False:
                                continue
                            # endif

                            iReadOutsPerExp = xTrgRsExp.GetReadOutsPerExp()
                            lTrgRoRows = xTrgRsExp.GetExpRowList()

                            iSrcTopOffset = xSrcRsExp.GetExpLineTopOffset()
                            lSrcRoRows = [x - iSrcTopOffset for x in lTrgRoRows]

                            print(
                                "Exposure {}: {}, [{}:{}] -> [{}:{}]".format(
                                    iRsIdx,
                                    iSrcTopOffset,
                                    lSrcRoRows[0],
                                    lSrcRoRows[-1],
                                    lTrgRoRows[0],
                                    lTrgRoRows[-1],
                                )
                            )

                            imgTrg = lImgTrg[iRsIdx]

                            if sContentType == "image":
                                imgTrg[lTrgRoRows, :, :] += imgSrc[lSrcRoRows, :, :] / iReadOutsPerExp

                            elif sContentType == "depth":
                                imgSrcDepthData = imgSrc[lSrcRoRows, :, 0]
                                imgSrcDepthValid = (imgSrcDepthData < 1e9) & np.isfinite(imgSrcDepthData)
                                imgSrcDepthValidF = imgSrcDepthValid.astype(np.float64)
                                np.nan_to_num(
                                    imgSrcDepthData,
                                    copy=False,
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                imgSrcDepthMasked = (imgSrcDepthData * imgSrcDepthValid).astype(np.float64)

                                imgTrg[lTrgRoRows, :, 0] += imgSrcDepthMasked
                                imgTrg[lTrgRoRows, :, 1] += np.square(imgSrcDepthMasked)
                                imgTrg[lTrgRoRows, :, 2] += imgSrcDepthValidF

                            elif sContentType == "pos3d":
                                imgSrcPos3dData = imgSrc[lSrcRoRows, :, :]
                                iSrcChnl: int = imgSrc.shape[2]
                                aOrig = np.zeros(iSrcChnl)
                                aOrig[[self.iChX, self.iChY, self.iChZ]] = np.array(dicCamera.get("lOrigin"))

                                imgMask = np.logical_or(
                                    np.abs(imgSrcPos3dData[:, :, self.iChX]) > 0.0,
                                    np.abs(imgSrcPos3dData[:, :, self.iChY]) > 0.0,
                                )
                                imgMask = np.logical_or(imgMask, np.abs(imgSrcPos3dData[:, :, self.iChZ]) > 0.0)
                                imgMask = np.logical_and(imgMask, np.abs(imgSrcPos3dData[:, :, self.iChX]) < 9e5)

                                imgTrgValid = np.zeros_like(imgMask)
                                imgTrgValid[imgMask] = 1.0

                                imgTrgRel = np.subtract(imgSrcPos3dData, aOrig)
                                imgTrgRad2 = np.sum(np.square(imgTrgRel), axis=2)
                                imgTrgRad = np.sqrt(imgTrgRad2)

                                imgTrgRad2[~imgMask] = 0.0
                                imgTrgRad[~imgMask] = 0.0

                                imgTrg[lTrgRoRows, :, 0] += imgTrgRad
                                imgTrg[lTrgRoRows, :, 1] += imgTrgRad2
                                imgTrg[lTrgRoRows, :, 2] += imgTrgValid

                            else:
                                raise Exception("Content type '{0}' not supported.".format(sContentType))
                            # endif
                        # endif
                    # endif

                    iRoLoopIdx += 1
                    # if iRoLoopIdx > 5:
                    #     break
                    # endif

                    # Next read out step
                    for iRsIdx, xTrgRsExp in enumerate(lTrgRsExp):
                        lValidRsLoop[iRsIdx] = xTrgRsExp.StepReadOutLoop()
                    # endfor

                    bSrcValid = True
                    iTrgReadOutStepIdx += 1
                    if iTrgReadOutStepIdx == iTrgPerSrcReadOutSteps:
                        iTrgReadOutStepIdx = 0
                        bSrcValid = xSrcRsExp.StepReadOutLoop()
                    # endif

                    # print("{}: {}".format(bSrcValid, str(lValidRsLoop)))
                    if not bSrcValid or not any(lValidRsLoop):
                        break
                    # endif
                # endwhile

                print("")

                bIsSingleExp = len(lTrgRsExp) == 1
                for iRsIdx, xTrgRsExp in enumerate(lTrgRsExp):
                    sFilename = sFrameName + ".exr"
                    sFpTrg = os.path.join(lPathTrgMain[iRsIdx], sFilename)
                    if bIsSingleExp is True:
                        print("> Writing constructed image '{0}'...".format(sFilename))
                    else:
                        print("> Writing constructed image for expose {}: '{}'...".format(iRsIdx + 1, sFilename))
                    # endif
                    print("> {0}".format(sFpTrg))
                    print("")

                    if self.bDoEval:
                        imgTrg = lImgTrg[iRsIdx]

                        if sContentType == "depth" or sContentType == "pos3d":
                            imgTrgDepthEval = np.zeros(lTrgImgSize, np.float64)
                            imgTrgDepthValid = imgTrg[:, :, 2] > 0.0
                            # imgTrgDepthValidF = imgTrgDepthValid.astype(np.float)

                            # imgTrgDepth[:, :, 0] *= imgTrgDepthValidF
                            # imgTrgDepth[:, :, 1] *= imgTrgDepthValidF

                            for i in range(0, 2):
                                imgTrg[:, :, i] = np.divide(
                                    imgTrg[:, :, i],
                                    imgTrg[:, :, 2],
                                    where=imgTrgDepthValid,
                                )
                                imgTrg[:, :, i] = np.where(imgTrgDepthValid, imgTrg[:, :, i], 0.0)
                            # endfor

                            imgSq = np.square(imgTrg[:, :, 0])
                            imgDiff = np.subtract(imgTrg[:, :, 1], imgSq)

                            imgDiffPos = imgDiff > 0.0
                            imgDiff = np.where(imgDiffPos, imgDiff, 0.0)
                            imgStdDev = np.sqrt(imgDiff, where=imgTrgDepthValid)
                            imgStdDev = np.where(imgTrgDepthValid, imgStdDev, 0.0)

                            imgTrgDepthEval[:, :, 2] = imgTrg[:, :, 0]
                            imgTrgDepthEval[:, :, 1] = imgStdDev
                            imgTrgDepthEval[:, :, 0] = np.where(imgTrgDepthValid, imgTrg[:, :, 2], 0.0)

                            cv2.imwrite(sFpTrg, imgTrgDepthEval.astype(np.float32))

                        elif sContentType == "image":
                            cv2.imwrite(sFpTrg, imgTrg)

                        else:
                            raise Exception("Content type '{0}' not supported".format(sContentType))
                        # endif
                    # endif
                # endfor

                print("> ... done")
                print("", flush=True)
            # endif Start Read Out

            iTrgFrame += self.iFrameStep
            iTrgFrameIdx += 1
        # endwhile iTrgFrame

    # enddef


# endclass
