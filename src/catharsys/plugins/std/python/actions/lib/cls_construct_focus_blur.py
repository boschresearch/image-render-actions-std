#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /do-tonemap.py
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

from catharsys.plugins.std.python.config.cls_construct_focus_blur_v1 import CConfigConstructFocusBlurModel1

print("Initializing CUDA...", flush=True)
from catharsys.plugins.std.cuda.cls_eval_focus_blur_model1 import CEvalFocusBlurModel1

print("done", flush=True)

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


################################################################################
class CConstructFocusBlur:

    ################################################################################

    ################################################################################

    ################################################################################
    # Properties
    @property
    def dicTrgImgType(self) -> dict:
        return self._dicTrgImgType

    # enddef

    ################################################################################
    def __init__(self):
        # Constants

        # Define output types and formats
        self._dicTrgImgType: dict = {
            "blur": {"sFolder": ".", "sExt": ".png"},
            # "preview": {"sFolder": "Preview", "sExt": ".png"},
            # "debug": {"sFolder": "_debug", "sExt": ".png"},
        }

        # Member variables
        self.xPrjCfg: CProjectConfig = None
        self.dicConfig: dict = None
        self.dicData: dict = None

        self.sPathTrgMain: str = None
        self.pathSrcDepth: Path = None
        self.pathSrcImage: Path = None
        self.dicPathTrgAct: dict = None
        self.dicActDtiToName: dict = None
        self.lActions: list = None
        self.iFrameFirst: int = None
        self.iFrameLast: int = None
        self.iFrameStep: int = None
        self.bDoProcess: bool = None
        self.bDoOverwrite: bool = None
        self.iDoProcess: int = None
        self.iDoOverwrite: int = None

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

        if len(imgX.shape) == 2:
            return imgX

        elif len(imgX.shape) == 3:
            if imgX.shape[2] == 1:
                return imgX
            elif imgX.shape[2] == 3:
                return imgX[:, :, [2, 1, 0]]
            elif imgX.shape[2] == 4:
                return imgX[:, :, [2, 1, 0, 3]]
            else:
                raise RuntimeError(
                    f"Expect image with 1, 3 or 4 channels, but given image has {(imgX.shape[2])}: {(_pathImage.as_posix())}"
                )
            # endif

        else:
            raise RuntimeError(
                f"Expect image as 2d or 3d array, but given image has {(len(imgX.shape))} dimensions: {(_pathImage.as_posix())}"
            )

        # endif

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
        sConstructFocusBlurDti = "blender/construct/focus-blur/*:1"

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

        lCMB = cathcfg.GetDataBlocksOfType(self.dicData, sConstructFocusBlurDti)
        if len(lCMB) == 0:
            raise Exception(
                "No label construction configuration of type compatible to '{0}' given".format(sConstructFocusBlurDti)
            )
        # endif
        dicCfgDti = cathcfg.CheckConfigType(lCMB[0], sConstructFocusBlurDti)
        sFocusBlurType = dicCfgDti["lCfgType"][4]
        xCfg = None
        if sFocusBlurType == "model1":
            xCfg = CConfigConstructFocusBlurModel1(lCMB[0])

            funcEvalInit = lambda tImageShape: CEvalFocusBlurModel1(
                _tiImageShape=tImageShape,
                _tiFilterRadiusXY=xCfg.tFilterRadiusXY,
                _tiStartXY=xCfg.tStartXY,
                _tiRangeXY=xCfg.tRangeXY,
            )

            funcEval = lambda xEvalFocusBlur, imgImage, imgDepth: xEvalFocusBlur.Eval(
                _imgImage=imgImage,
                _imgDepth=imgDepth,
                _fFocusDepth_mm=xCfg.fFocusDepth_mm,
                _fFocalLength_mm=xCfg.fFocalLength_mm,
                _fApertureDia_mm=xCfg.fApertureDia_mm,
                _fPixelPitch_mm=xCfg.fPixelPitch_mm,
                _fFocalPlanePos_mm=xCfg.fFocalPlanePos_mm,
                _fMMperDepthUnit=xCfg.fMMperDepthUnit,
            )

        else:
            raise RuntimeError(f"Unsupported focus blur type '{sFocusBlurType}'")
        # endif

        # Initialize variable which will contain class instance of flow eval algo.
        # This can only be initialized once the image size is known.
        xEvalFocusBlur = None

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

        self.pathSrcDepth = xCfg.pathDepth
        if not self.pathSrcDepth.is_absolute():
            # Get the path to the local position render
            self.pathSrcDepth = cathpath.MakeNormPath((sPathRenderAct, self.pathSrcDepth))
        # endif

        self.pathSrcImage = xCfg.pathImage
        if not self.pathSrcImage.is_absolute():
            self.pathSrcImage = cathpath.MakeNormPath((sPathRenderAct, self.pathSrcImage))
        # endif

        ###################################################################################
        print("\nDepth source main path: {0}".format(self.pathSrcDepth.as_posix()))
        print("\nImage source main path: {0}".format(self.pathSrcImage.as_posix()))
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

            iImageFrame: int = None

            iImageFrame = iTrgFrame

            print("")
            print(f"Start processing focus blur for frame {iImageFrame}...", flush=True)

            ###################################################################
            sTrgFrameName = "Frame_{0:04d}".format(iImageFrame)
            sFrameName = "Frame_{0:04d}".format(iImageFrame)
            sFileImgSrc = f"{sFrameName}{xCfg.sImageFileExt}"
            sFileDepthSrc = f"{sFrameName}.exr"

            pathFileImg = cathpath.MakeNormPath((self.pathSrcImage, sFileImgSrc))
            if not pathFileImg.exists():
                print(f"Image frame '{iImageFrame}' does not exist at {(pathFileImg.as_posix())}. Skipping...")
                continue
            # endif

            pathFileDepth = cathpath.MakeNormPath((self.pathSrcDepth, sFileDepthSrc))
            if not pathFileDepth.exists():
                print(f"Flow frame '{iImageFrame}' does not exist at {(pathFileDepth.as_posix())}. Skipping...")
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

            # Load images
            print("Loading images...", flush=True)
            imgImage = self._LoadImage(pathFileImg)
            if imgImage is None:
                continue
            # endif

            imgDepth = self._LoadImage(pathFileDepth)
            if imgDepth is None:
                continue
            # endif
            print("done.", flush=True)

            iImgRows, iImgCols = imgImage.shape[0:2]
            iDepthRows, iDepthCols = imgDepth.shape[0:2]

            if iDepthRows != iImgRows or iDepthCols != iImgCols:
                print(
                    "Error: image and depth have different sizes: "
                    f"[{iImgCols}, {iImgRows}] vs [{iDepthCols}, {iDepthRows}]"
                )
                continue
            # endif

            ################################################################################
            # Evaluate focus blur

            # Only initialize blur algo kernel once.
            # Assumes that all images have the same size.
            if xEvalFocusBlur is None:
                print("Compiling CUDA kernel...", flush=True)
                xEvalFocusBlur = funcEvalInit(imgImage.shape)
                print("done")
            # endif

            print("Evaluate motion blur...", flush=True)
            funcEval(xEvalFocusBlur, imgImage, imgDepth)
            print("done", flush=True)

            print("Saving motion blur image...", flush=True)
            sFpImgTrg: str = self.dicTrgImgType["blur"]["sFpImgTrg"]
            xEvalFocusBlur.SaveBlurImage(sFpImgTrg)
            print("done", flush=True)
        # endwhile iTrgFrame

    # enddef


# endclass
