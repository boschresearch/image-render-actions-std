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
import math
import numpy as np

# from datetime import datetime
from pathlib import Path

from catharsys.config.cls_project import CProjectConfig
from anybase import assertion
import catharsys.util.config as cathcfg
import catharsys.util.file as cathfile
import catharsys.util.path as cathpath
from catharsys.plugins.std.blender.config.cls_compositor import CConfigCompositor

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


################################################################################
class CTonemap:

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
        sImgRelPathDti = "blender/tonemap/input-rel-path:1"
        sImgTypDti = "blender/tonemap/input-id:1"
        sToneMapDti = "blender/tonemap/type/*:1"
        sRenderDti = "blender/render/output/image:1"
        sRenderOutputListDti = "blender/render/output-list:1"

        lRndTypeList = cathcfg.GetDataBlocksOfType(self.dicData, sRenderOutputListDti)
        if len(lRndTypeList) == 0:
            raise Exception(
                "No render output configuration of type compatible to '{0}' given".format(sRenderOutputListDti)
            )
        # endif
        dicRndOutList = lRndTypeList[0]
        lRndOutTypes = cathcfg.GetDictValue(dicRndOutList, "lOutputs", list)
        if lRndOutTypes is None:
            raise Exception("No render output types defined")
        # endif

        # Look for 'image' render output type
        dicRndOut = None
        for dicOut in lRndOutTypes:
            dicRes = cathcfg.CheckConfigType(dicOut, sRenderDti)
            if dicRes["bOK"] is True:
                dicRndOut = dicOut
                break
            # endif
        # endfor

        if dicRndOut is None:
            raise Exception("No render output type 'image' specified in configuration")
        # endif

        dicComp = dicRndOut.get("mCompositor")
        cathcfg.AssertConfigType(dicComp, "/catharsys/blender/compositor:1")
        xComp = CConfigCompositor(xPrjCfg=self.xPrjCfg, dicData=dicComp)

        lImgRelPath = cathcfg.GetDataBlocksOfType(self.dicData, sImgRelPathDti)
        if len(lImgRelPath) == 0:
            raise Exception("No relative path configuration of type compatible to '{0}' given.".format(sImgRelPathDti))
        # endif
        sImageRelPath = lImgRelPath[0]

        lImgFolder = cathcfg.GetDataBlocksOfType(self.dicData, sImgTypDti)
        if len(lImgFolder) == 0:
            raise Exception("No image type configuration of type compatible to '{0}' given.".format(sImgTypDti))
        # endif
        sImageFolder = lImgFolder[0]

        lToneMap = cathcfg.GetDataBlocksOfType(self.dicData, sToneMapDti)
        if len(lToneMap) == 0:
            raise Exception("No tonemap configuration of type compatible to '{0}' given.".format(sToneMapDti))
        # endif
        dicToneMap: dict = lToneMap[0]

        # Get compositor file format for given image type
        lCompFo = xComp.GetOutputsByFolderName()
        # print(lCompFo)
        # print(sImageFolder)

        lImageFolderFo = lCompFo.get(sImageFolder, [])
        if len(lImageFolderFo) == 0:
            raise Exception("Compositor configuration does not contain an output to folder '{0}'.".format(sImageFolder))
        elif len(lImageFolderFo) > 1:
            raise Exception(
                "Compositor configuration contains more than one output to the folder '{0}'.".format(sImageFolder)
            )
        # endif
        dicImageFolderFo = lImageFolderFo[0]

        # print(lImageFolderFo)

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

        if len(self.lActions) < 2:
            raise Exception("Tonemap action has no parent action.")
        # endif
        sParentAction = self.lActions[-2]
        self.pathSrcMain = cathpath.MakeNormPath((self.dicPathTrgAct.get(sParentAction), sImageRelPath))

        print("Image source main path: {0}".format(self.pathSrcMain.as_posix()))
        print("Image target main path: {0}".format(self.sPathTrgMain))
        print("First rendered frame: {0}".format(self.iFrameFirst))
        print("Last rendered frame: {0}".format(self.iFrameLast))
        print("Frame step: {0}".format(self.iFrameStep))
        print("Do process: {0}".format(self.bDoProcess))

        ###################################################################################
        # Prepare image capture data for processing
        lSupportedToneMaps = ["linear", "gamma", "log", "sub-gauss"]

        sToneMapType = cathcfg.SplitDti(dicToneMap.get("sDTI")).get("lType")[4]
        print("Tone-Map type: {0}".format(sToneMapType))
        if sToneMapType not in lSupportedToneMaps:
            raise Exception("Unsupported tone map type '{0}'".format(sToneMapType))
        # endif

        sToneMapId = cathcfg.GetDictValue(dicToneMap, "sId", str, sWhere=f"tonemap configuration '{sToneMapType}'")

        sFolderImgSrc = cathcfg.GetDictValue(dicImageFolderFo, "sFolder", str, sWhere="compositor configuration")
        pathImageSrc: Path = self.pathSrcMain / sFolderImgSrc
        print("{1}: Image source path: {0}".format(pathImageSrc.as_posix(), sToneMapId))
        if not pathImageSrc.exists():
            raise Exception(
                "{2}: Source image path for image type '{0}' does not exist: {1}".format(
                    sImageFolder, pathImageSrc.as_posix(), sToneMapId
                )
            )
        # endif

        # sPathImageTrg = self.pathSrcMain.as_posix() #os.path.join(sPathImageSrc, sToneMapId)
        # cathpath.CreateDir(sPathImageTrg)
        pathImageTrg: Path = self.pathSrcMain / sToneMapId
        pathImageTrg.mkdir(parents=True, exist_ok=True)
        print(
            "{2}: Image target path for image type '{0}': {1}".format(sImageFolder, pathImageTrg.as_posix(), sToneMapId)
        )

        # Store the collected construction parameters in JSON file with constructed images
        pathToneMapOut = pathImageTrg / "ToneMap.json"
        cathfile.SaveJson(pathToneMapOut, dicToneMap, iIndent=4)

        # Source image extension
        sSrcImgExt = dicImageFolderFo.get("sFileExt")

        # Source image type override
        sSrcImgType = dicToneMap.get("sSrcImgType")
        if sSrcImgType is not None:
            if sSrcImgType == "jpg":
                sSrcImgExt = ".jpg"
            elif sSrcImgType == "png":
                sSrcImgExt = ".png"
            elif sSrcImgType == "exr":
                sSrcImgExt = ".exr"
            else:
                raise RuntimeError(f"Unsupported source image type '{sSrcImgType}'")
            # endif
        # endif

        # Loop over frames
        iTrgFrame = self.iFrameFirst
        iTrgFrameIdx = 0
        iTrgFrameCnt = int(math.floor((self.iFrameLast - self.iFrameFirst) / self.iFrameStep)) + 1

        while iTrgFrame <= self.iFrameLast:

            print("")
            print(
                "{1}: Start processing frame {0}...".format(iTrgFrame, sToneMapId),
                flush=True,
            )

            sFrameName = "Frame_{0:04d}".format(iTrgFrame)
            sFileImgSrc = "{0}{1}".format(sFrameName, sSrcImgExt)

            pathImgSrcFile = pathImageSrc / sFileImgSrc
            if not pathImgSrcFile.exists():
                print("{1}: Frame '{0}' does not exist. Skipping...".format(sFileImgSrc, sToneMapId))
                iTrgFrame += self.iFrameStep
                iTrgFrameIdx += 1
                continue
            # endif

            # Check if target images already exist
            bTrgImgMissing = False

            lTrgImgFormat = dicToneMap.get("lTrgImgFormat")
            for dicTrgImgFormat in lTrgImgFormat:
                sTrgImgType: str = dicTrgImgFormat.get("sType")
                sFileImgTrg: str = sFrameName + "." + sTrgImgType
                pathImgTrgFile = pathImageTrg / sFileImgTrg
                if not pathImgTrgFile.exists():
                    bTrgImgMissing = True
                elif self.bDoOverwrite:
                    print("Removing file due to overwrite flag: {0}".format(pathImgTrgFile.as_posix()))
                    pathImgTrgFile.unlink()
                # endif
            # endfor

            if not bTrgImgMissing and not self.bDoOverwrite:
                print("{0}: frame '{1}' already tonemapped. Skipping...".format(sToneMapId, sFrameName))
                iTrgFrame += self.iFrameStep
                iTrgFrameIdx += 1
                continue
            # endif

            # Load source image
            # print("Loading image: {0}".format(sFpImgSrc))
            imgSrcImg = cv2.imread(
                pathImgSrcFile.as_posix(),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
            )

            typSrcImg = imgSrcImg.dtype
            # fSrcMin = np.amin(imgSrcImg)
            # fSrcMax = np.amax(imgSrcImg)
            imgR = imgSrcImg.astype(np.float64)

            if np.issubdtype(typSrcImg, np.integer):
                iMin = np.iinfo(typSrcImg).min
                iMax = np.iinfo(typSrcImg).max
                imgR -= iMin
                imgR /= iMax - iMin
            # endif

            fMin = np.amin(imgR)
            fMax = np.amax(imgR)
            # print(f"{fMin} -> {fMax}")

            # If black and white level are not given, use min and max of image
            fBlack = dicToneMap.get("dSrcBlackValue", dicToneMap.get("fSrcBlackValue", fMin))
            fWhite = dicToneMap.get("dSrcWhiteValue", dicToneMap.get("fSrcWhiteValue", fMax))
            print(f"{sToneMapId}: Using black/white levels: {fBlack} / {fWhite}")

            if sToneMapType == "linear":
                dScale = 1.0 / (fWhite - fBlack)

                imgR -= fBlack
                imgR *= dScale
                imgR = np.where(imgR < 0.0, 0.0, imgR)
                imgR = np.where(imgR > 1.0, 1.0, imgR)

            elif sToneMapType == "gamma":
                fGamma = dicToneMap.get("dGamma", dicToneMap.get("fGamma"))
                if fGamma is None:
                    raise RuntimeError("Element 'fGamma' not specified in configuration")
                # endif

                dScale = 1.0 / (fWhite - fBlack)

                imgR -= fBlack
                imgR *= dScale
                imgR = np.where(imgR < 0.0, 0.0, imgR)
                imgR = np.where(imgR > 1.0, 1.0, imgR)

                imgR = np.power(imgR, fGamma)

            elif sToneMapType == "log":
                imgR -= fBlack
                imgR = np.where(imgR < 0.0, 0.0, imgR)
                imgR = np.log(imgR + 1.0)

                dLogWhite = math.log(fWhite + 1.0)
                imgR /= dLogWhite

                imgR = np.where(imgR > 1.0, 1.0, imgR)
            # endif tonemap type

            # write result image
            lTrgImgFormat = dicToneMap.get("lTrgImgFormat")
            for dicTrgImgFormat in lTrgImgFormat:
                sTrgImgType = dicTrgImgFormat.get("sType")
                iTrgBitDepth = dicTrgImgFormat.get("iBitDepth")
                sColorType = dicTrgImgFormat.get("sColorType", "rgb")

                if sTrgImgType == "png":
                    if iTrgBitDepth < 1 or iTrgBitDepth > 16:
                        raise Exception(
                            "{2}: Target bit depth of {0}bit is not supported by file type '{1}'.".format(
                                iTrgBitDepth, sTrgImgType, sToneMapId
                            )
                        )
                    # endif

                    sFileImgTrg: str = sFrameName + "." + sTrgImgType
                    pathImgTrgFile = pathImageTrg / sFileImgTrg
                    if pathImgTrgFile.exists():
                        print(
                            "{0}: image '{1}' already exist. Skipping...".format(sToneMapId, pathImgTrgFile.as_posix())
                        )
                        continue
                    # endif

                    if sColorType == "gray":
                        # Convert to gray values
                        lImgSize = imgR.shape
                        imgRF = np.sum(imgR, axis=2) / lImgSize[2]

                    elif sColorType == "rgb":
                        imgRF = imgR
                    # endif

                    dMaxValue = math.pow(2.0, iTrgBitDepth) - 1.0
                    imgRF = np.around(imgRF * dMaxValue)

                    if iTrgBitDepth <= 8:
                        imgRF = imgRF.astype(np.uint8)
                    else:
                        imgRF = imgRF.astype(np.uint16)
                    # endif target bit depth

                    print(
                        "{2}: Writing format '{0}' with bit depth '{1}'...".format(
                            sTrgImgType, iTrgBitDepth, sToneMapId
                        ),
                        flush=True,
                    )
                    if self.bDoProcess:
                        cv2.imwrite(pathImgTrgFile.as_posix(), imgRF)
                    # endif

                elif sTrgImgType == "pgm":
                    if iTrgBitDepth < 1 or iTrgBitDepth > 16:
                        raise Exception(
                            "{2}: Target bit depth of {0}bit is not supported by file type '{1}'.".format(
                                iTrgBitDepth, sTrgImgType, sToneMapId
                            )
                        )
                    # endif

                    sFileImgTrg: str = sFrameName + "." + sTrgImgType
                    pathImgTrgFile = pathImageTrg / sFileImgTrg
                    if pathImgTrgFile.exists():
                        print("{0}: image '{1}' already exist. Skipping...".format(sToneMapId, sFileImgTrg))
                        continue
                    # endif

                    # Convert to gray values
                    lImgSize = imgR.shape
                    imgRF = np.sum(imgR, axis=2) / lImgSize[2]

                    dMaxValue = math.pow(2.0, iTrgBitDepth) - 1.0
                    imgRF = np.around(imgRF * dMaxValue)

                    if iTrgBitDepth <= 8:
                        imgRF = imgRF.astype(np.uint8)
                    else:
                        imgRF = imgRF.astype(np.uint16)
                    # endif target bit depth

                    print(
                        "{2}: Writing format '{0}' with bit depth '{1}'...".format(
                            sTrgImgType, iTrgBitDepth, sToneMapId
                        ),
                        flush=True,
                    )
                    if self.bDoProcess:
                        cv2.imwrite(pathImgTrgFile, imgRF)
                    # endif

                elif sTrgImgType == "jpg":
                    if iTrgBitDepth < 1 or iTrgBitDepth > 8:
                        raise Exception(
                            "{2}: Target bit depth of {0}bit is not supported by file type '{1}'.".format(
                                iTrgBitDepth, sTrgImgType, sToneMapId
                            )
                        )
                    # endif

                    sFileImgTrg: str = sFrameName + "." + sTrgImgType
                    pathImgTrgFile = pathImageTrg / sFileImgTrg
                    if pathImgTrgFile.exists():
                        print("{0}: image '{1}' already exist. Skipping...".format(sToneMapId, sFileImgTrg))
                        continue
                    # endif

                    dMaxValue = math.pow(2.0, iTrgBitDepth) - 1.0
                    imgRF = np.around(imgR * dMaxValue)
                    imgRF = imgRF.astype(np.uint8)

                    print(
                        "{2}: Writing format '{0}' with bit depth '{1}'...".format(
                            sTrgImgType, iTrgBitDepth, sToneMapId
                        ),
                        flush=True,
                    )
                    if self.bDoProcess:
                        cv2.imwrite(pathImgTrgFile.as_posix(), imgRF)
                    # endif

                else:
                    print(
                        "{1}: WARNING: Image format type '{0}' not supported".format(sTrgImgType, sToneMapId),
                        flush=True,
                    )
                # endif target image type

            # endfor target image format
            print("")

            iTrgFrame += self.iFrameStep
            iTrgFrameIdx += 1
        # endwhile iTrgFrame

    # enddef


# endclass
