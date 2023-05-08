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

# from anybase import assertion
from anybase.cls_any_error import CAnyError_Message

# from anycam.obj import camera as camviewfactory

from catharsys.config.cls_project import CProjectConfig
import catharsys.util.config as cathcfg

# import catharsys.util.file as cathfile
import catharsys.util.path as cathpath
import catharsys.plugins.std

from catharsys.plugins.std.python.config.cls_anytruth_construct_flow_v1 import CAnytruthConstructFlow1

print("Initializing CUDA...", flush=True)
from catharsys.plugins.std.cuda.cls_eval_flow_ground_truth import CEvalFlowGroundTruth

print("done", flush=True)

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


################################################################################
class CConstructFlow:

    ################################################################################
    # Constants
    _iChX: int = 2
    _iChY: int = 1
    _iChZ: int = 0

    # Define output types and formats
    _dicTrgImgType: dict = {
        "flow": {"sFolder": ".", "sExt": ".exr"},
        # "preview": {"sFolder": "Preview", "sExt": ".png"},
        # "debug": {"sFolder": "_debug", "sExt": ".png"},
    }

    ################################################################################
    # Member variables
    xPrjCfg: CProjectConfig = None
    dicConfig: dict = None
    dicData: dict = None

    sPathTrgMain: str = None
    pathSrcLocalPos3d: Path = None
    pathSrcObjIdx: Path = None
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
    def _GetRndOutDict(self, *, _sSubType: str, _lRndOutTypes: list[dict]) -> dict:

        # Look for 'anytruth/pos3d' render output type
        dicRndOut = None
        for dicOut in _lRndOutTypes:
            dicRes = cathcfg.CheckConfigType(dicOut, f"/catharsys/blender/render/output/{_sSubType}")
            if dicRes["bOK"] is True:
                dicRndOut = dicOut
                break
            # endif
        # endfor

        if dicRndOut is None:
            raise Exception(f"No render output type '{_sSubType}' specified in configuration")
        # endif

        return dicRndOut

    # enddef

    ################################################################################
    def _LoadImage(self, _pathImage: Path) -> np.ndarray:

        imgX = cv2.imread(
            _pathImage.as_posix(),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
        )

        if imgX is None:
            print("Error loading image: {0}".format(_pathImage.as_posix()))
        # endif

        return imgX

    # enddef

    ################################################################################
    def Process(self, _xPrjCfg: CProjectConfig, _dicCfg: dict, **kwargs):
        # assertion.FuncArgTypes()

        self.xPrjCfg = _xPrjCfg

        sWhere: str = "action configuration"
        self.dicConfig = cathcfg.GetDictValue(_dicCfg, "mConfig", dict, sWhere=sWhere)
        self.dicData = cathcfg.GetDictValue(self.dicConfig, "mData", dict, sWhere=sWhere)

        # Define expected type names
        sRenderTypeListDti = "blender/render/output-list:1"
        sConstructFlowDti = "blender/anytruth/construct/flow:1"

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

        # dicRndOutLocalPos3d = self._GetRndOutDict(_sSubType="anytruth/local-pos3d:1", _lRndOutTypes=lRndOutTypes)
        # dicRndOutObjIdx = self._GetRndOutDict(_sSubType="anytruth/object-idx:1", _lRndOutTypes=lRndOutTypes)

        lCFT = cathcfg.GetDataBlocksOfType(self.dicData, sConstructFlowDti)
        if len(lCFT) == 0:
            raise Exception(
                "No label construction configuration of type compatible to '{0}' given".format(sConstructFlowDti)
            )
        # endif
        xCFT = CAnytruthConstructFlow1(lCFT[0])

        # Initialize variable which will contain class instance of flow eval algo.
        # This can only be initialized once the image size is known.
        xEvalFlow: CEvalFlowGroundTruth = None

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

        # Get the path to the local position render
        self.pathSrcLocalPos3d = cathpath.MakeNormPath((sPathRenderAct, "AT_LocalPos3d_Raw"))
        # Get the path to the object index render
        self.pathSrcObjIdx = cathpath.MakeNormPath((sPathRenderAct, "AT_ObjectIdx_Raw"))

        ###################################################################################
        print("\nLocal pos. 3d source main path: {0}".format(self.pathSrcLocalPos3d.as_posix()))
        print("\nObject index source main path: {0}".format(self.pathSrcObjIdx.as_posix()))
        print("\nMain output path: {0}".format(self.sPathTrgMain))
        print("\nFirst rendered frame: {0}".format(self.iFrameFirst))
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

        ###################################################################################
        # Loop over frames
        iTrgFrame = self.iFrameFirst - self.iFrameStep
        iTrgFrameIdx = -1
        # iTrgFrameCnt = int(math.floor((self.iFrameLast - self.iFrameFirst) / self.iFrameStep)) + 1

        # Do not evaluate flow for last frame "self.iFrameLast", because we assume
        # that this is the last rendered frame for which no forward flow can be calculated.
        while (iTrgFrame + self.iFrameStep) < self.iFrameLast:

            # increment indices here to enable 'continue'
            # without need to increment at each continue command
            iTrgFrame += self.iFrameStep
            iTrgFrameIdx += 1

            iFlowFrame1: int = None
            iFlowFrame2: int = None

            if xCFT.iFrameDelta > 0:
                iFlowFrame1 = iTrgFrame
                iFlowFrame2 = iTrgFrame + xCFT.iFrameDelta
            elif xCFT.iFrameDelta < 0:
                iFlowFrame1 = iTrgFrame - xCFT.iFrameDelta
                iFlowFrame2 = iTrgFrame
            else:
                raise CAnyError_Message(sMsg="Flow frame delta must not be zero.")
            # endif

            print("")
            print(f"Start processing flow for frames {iFlowFrame1} -> {iFlowFrame2}...", flush=True)

            ###################################################################
            sTrgFrameName = "Frame_{0:04d}".format(iFlowFrame1)
            sFrameName1 = "Frame_{0:04d}".format(iFlowFrame1)
            sFrameName2 = "Frame_{0:04d}".format(iFlowFrame2)
            sFileImgSrc1 = "{0}.exr".format(sFrameName1)
            sFileImgSrc2 = "{0}.exr".format(sFrameName2)

            pathImgLocalPos3d1 = cathpath.MakeNormPath((self.pathSrcLocalPos3d, sFileImgSrc1))
            if not pathImgLocalPos3d1.exists():
                print(
                    f"Local pos. 3d frame '{iFlowFrame1}' does not exist at {(pathImgLocalPos3d1.as_posix())}. Skipping..."
                )
                continue
            # endif

            pathImgObjIdx1 = cathpath.MakeNormPath((self.pathSrcObjIdx, sFileImgSrc1))
            if not pathImgObjIdx1.exists():
                print(
                    f"Object index frame '{iFlowFrame1}' does not exist at {(pathImgObjIdx1.as_posix())}. Skipping..."
                )
                continue
            # endif

            pathImgLocalPos3d2 = cathpath.MakeNormPath((self.pathSrcLocalPos3d, sFileImgSrc2))
            if not pathImgLocalPos3d2.exists():
                print(
                    f"Local pos. 3d frame '{iFlowFrame2}' does not exist at {(pathImgLocalPos3d2.as_posix())}. Skipping..."
                )
                continue
            # endif

            pathImgObjIdx2 = cathpath.MakeNormPath((self.pathSrcObjIdx, sFileImgSrc2))
            if not pathImgObjIdx2.exists():
                print(
                    f"Object index frame '{iFlowFrame2}' does not exist at {(pathImgObjIdx2.as_posix())}. Skipping..."
                )
                continue
            # endif

            # Check if target images already exist
            bTrgImgMissing = False

            for sTrgImgType in self.dicTrgImgType:
                dicImgType = self.dicTrgImgType.get(sTrgImgType)
                pathImgTrg = Path(dicImgType.get("sPathImgTrg"))
                sFileImgTrg = "{0}{1}".format(sTrgFrameName, dicImgType.get("sExt"))
                pathImgTrg = pathImgTrg / sFileImgTrg
                dicImgType["sFileImgTrg"] = sFileImgTrg
                dicImgType["sFpImgTrg"] = pathImgTrg.as_posix()

                if not pathImgTrg.exists():
                    bTrgImgMissing = True
                elif self.bDoOverwrite:
                    print(f"Removing file due to overwrite flag: {(pathImgTrg.as_posix())}")
                    pathImgTrg.unlink()
                # endif
            # endfor

            if not bTrgImgMissing and not self.bDoOverwrite:
                print(f"Frame '{sTrgFrameName}' already processed. Skipping...")
                continue
            # endif

            # Load local pos 3d image
            print("Loading images...", flush=True)
            imgLocalPos3d1 = self._LoadImage(pathImgLocalPos3d1)
            if imgLocalPos3d1 is None:
                continue
            # endif

            imgLocalPos3d2 = self._LoadImage(pathImgLocalPos3d2)
            if imgLocalPos3d2 is None:
                continue
            # endif

            imgObjIdx1 = self._LoadImage(pathImgObjIdx1)
            if imgObjIdx1 is None:
                continue
            # endif

            imgObjIdx2 = self._LoadImage(pathImgObjIdx2)
            if imgObjIdx2 is None:
                continue
            # endif
            print("done.", flush=True)

            iPosRows, iPosCols, iPosChnl = imgLocalPos3d1.shape
            iObjIdxRows, iObjIdxCols, iObjIdxChnl = imgObjIdx1.shape

            if iObjIdxRows != iPosRows or iObjIdxCols != iPosCols:
                print(
                    "Error: local 3d position and object index images have different sizes: "
                    f"[{iPosCols}, {iPosRows}] vs [{iObjIdxCols}, {iObjIdxRows}]"
                )
                continue
            # endif

            if imgLocalPos3d1.shape != imgLocalPos3d2.shape:
                print(
                    f"Error: local position images for frames {iFlowFrame1} and {iFlowFrame2} have different sizes: "
                    f"{imgLocalPos3d1.shape} vs {imgLocalPos3d2.shape}"
                )
                continue
            # endif

            if imgObjIdx1.shape != imgObjIdx2.shape:
                print(
                    f"Error: object index images for frames {iFlowFrame1} and {iFlowFrame2} have different sizes: "
                    f"{imgObjIdx1.shape} vs {imgObjIdx2.shape}"
                )
                continue
            # endif

            ################################################################################
            # Evaluate flow

            # Only initialize flow algo kernel once.
            # Assumes that all images have the same size.
            if xEvalFlow is None:
                print("Compiling CUDA kernel...", flush=True)
                xEvalFlow = CEvalFlowGroundTruth(
                    _tiImageShape=imgLocalPos3d1.shape,
                    _tiSearchRadiusXY=xCFT.tSearchRadiusXY,
                    _tiStartXY=xCFT.tStartXY,
                    _tiRangeXY=xCFT.tRangeXY,
                )
                print("done")
            # endif

            print("Evaluate flow...", flush=True)
            xEvalFlow.Eval(
                _imgPos1=imgLocalPos3d1,
                _imgPos2=imgLocalPos3d2,
                _imgObjIdx1=imgObjIdx1,
                _imgObjIdx2=imgObjIdx2,
                _iChObjIdx=2,
                _iChInstId=0,
                _bSphericalCS=False,
                _tRgbIdx=(2, 1, 0),
            )
            print("done", flush=True)

            print("Saving flow image...", flush=True)
            sFpImgTrg: str = self.dicTrgImgType["flow"]["sFpImgTrg"]
            xEvalFlow.SaveFlowImage(sFpImgTrg)
            print("done", flush=True)
        # endwhile iTrgFrame

    # enddef


# endclass
