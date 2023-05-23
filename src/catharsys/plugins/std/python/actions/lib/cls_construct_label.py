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
import re
import numpy as np
import copy
from pathlib import Path


# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import anybase.config
from anybase.cls_any_error import CAnyError_Message

from anybase import assertion
from anycam.obj import camera as camviewfactory

from catharsys.config.cls_project import CProjectConfig
import catharsys.util.config as cathcfg
import catharsys.util.file as cathfile
import catharsys.util.path as cathpath


################################################################################
class CConstructLabel:
    ################################################################################
    def __init__(self):
        pass

    # enddef

    ################################################################################
    # Constants
    _iTrgChType: int = 2
    _iTrgChInst: int = 1
    _iTrgChShInst: int = 0
    _iChType: int = 2
    _iChShType: int = 1
    _iChNorm: int = 0

    # Define output types and formats
    _dicTrgImgType: dict = {
        "label-image": {"sFolder": "SemSeg", "sExt": ".png"},
        "label-data": {"sFolder": "Data", "sExt": ".json"},
        "preview": {"sFolder": "Preview", "sExt": ".png"},
        "debug": {"sFolder": "_debug", "sExt": ".png"},
    }

    _reAtBone: re.Pattern = re.compile(r"^AT\.Label;(?P<skel>.+);(?P<bone>.+)")

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
    def iTrgChType(self) -> int:
        return self._iTrgChType

    # enddef

    @property
    def iTrgChInst(self) -> int:
        return self._iTrgChInst

    # enddef

    @property
    def iTrgChShInst(self) -> int:
        return self._iTrgChShInst

    # enddef

    @property
    def iChType(self) -> int:
        return self._iChType

    # enddef

    @property
    def iChShType(self) -> int:
        return self._iChShType

    # enddef

    @property
    def iChNorm(self) -> int:
        return self._iChNorm

    # enddef

    @property
    def dicTrgImgType(self) -> dict:
        return self._dicTrgImgType

    # enddef

    @property
    def reAtBone(self) -> re.Pattern:
        return self._reAtBone

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
        sConstructLabelDti = "blender/anytruth/construct/label:2"

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

        # Look for 'anytruth/label' render output type
        dicRndOut = None
        for dicOut in lRndOutTypes:
            dicRes = cathcfg.CheckConfigType(dicOut, "/catharsys/blender/render/output/anytruth/label:1")
            if dicRes["bOK"] is True:
                dicRndOut = dicOut
                break
            # endif
        # endfor

        if dicRndOut is None:
            raise Exception("No render output type 'anytruth/label' specified in configuration")
        # endif

        lCLT = cathcfg.GetDataBlocksOfType(self.dicData, sConstructLabelDti)
        if len(lCLT) == 0:
            raise Exception(
                "No label construction configuration " "of type compatible to '{0}' given".format(sConstructLabelDti)
            )
        # endif
        dicCLT = lCLT[0]

        # Construct regular expressions from box types
        # dicProcess = dicCLT.get("mProcess")

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
        self.pathSrcMain = cathpath.MakeNormPath((sPathRenderAct, "AT_Label_Raw"))

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
        iTrgFrame = self.iFrameFirst
        iTrgFrameIdx = 0
        # iTrgFrameCnt = int(math.floor((self.iFrameLast - self.iFrameFirst) / self.iFrameStep)) + 1

        ###################################################################
        # Loop over frames
        while iTrgFrame <= self.iFrameLast:
            print("")
            print("Start processing frame {0}...".format(iTrgFrame), flush=True)

            ###################################################################
            # Get label info
            try:
                dicFrameLabelSemSeg = self._ReadLabelConfig(iTrgFrame)
            except Exception as xEx:
                print("Error reading label config for frame {0}:\n{1}".format(iTrgFrame, str(xEx)))
                iTrgFrame += self.iFrameStep
                iTrgFrameIdx += 1
                continue
            # endtry

            # Check if target images already exist
            bTrgImgMissing = False

            ###################################################################
            # Loop over all target image types and set their target filenames
            sFrameName = self._GetFrameName(iTrgFrame)

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
                iTrgFrame += self.iFrameStep
                iTrgFrameIdx += 1
                continue
            # endif

            ################################################################################
            # Process the raw SemSeg label image
            dicResult = self._ProcessImgRawSemSeg(iTrgFrame=iTrgFrame, dicFrameLabelSemSeg=dicFrameLabelSemSeg)
            if dicResult["bOK"] is False:
                print(dicResult["sMsg"])
                iTrgFrame += self.iFrameStep
                iTrgFrameIdx += 1
                continue
            # endif

            imgTrgPreview = dicResult["imgTrgPreview"]
            imgTrgLabel = dicResult["imgTrgLabel"]

            ################################################################################
            # Create camera for projection of 3d data points
            try:
                xView = camviewfactory.CreateCameraView(
                    dicFrameLabelSemSeg.get("mCamera"), _xCamDataPath=self.pathSrcMain
                )
            except Exception as xEx:
                raise CAnyError_Message(sMsg="Error creating camera view", xChildEx=xEx)
            # endtry

            dicCamView = None
            if xView is not None and hasattr(xView, "GetDataDict"):
                dicCamView = xView.GetDataDict()
            # endif

            if xView is None or not hasattr(xView, "ProjectToImage"):
                xView = None
            # endif

            if xView is not None:
                iImgRows, iImgCols, iImgChnl = imgTrgLabel.shape
                if xView.iPixCntX != iImgCols or xView.iPixCntY != iImgRows:
                    raise Exception(
                        "Label image resolution ({}x{}) does not agree with camera resolution ({}x{})".format(
                            iImgCols, iImgRows, xView.iPixCntX, xView.iPixCntY
                        )
                    )
                # endif
            # endif

            ################################################################################
            # Process Skeletons
            self._ProcessSkeletons(
                xView=xView,
                imgTrgPreview=imgTrgPreview,
                imgTrgLabel=imgTrgLabel,
                dicFrameLabelSemSeg=dicFrameLabelSemSeg,
                dicCLT=dicCLT,
            )

            ################################################################################
            # Project 3d label data to image, if possible
            self._ProjectLabelDataToImage(
                xView=xView,
                imgTrgPreview=imgTrgPreview,
                dicFrameLabelSemSeg=dicFrameLabelSemSeg,
                dicCLT=dicCLT,
            )

            ################################################################################
            # Process 2D-Boxes
            self._ProcessIdealInstanceBoxes2d(
                imgTrgPreview=imgTrgPreview,
                dicFrameLabelSemSeg=dicFrameLabelSemSeg,
                dicCLT=dicCLT,
            )

            ################################################################################
            # Find 2d boxes
            self._FindSemSegInstanceBoxes2d(
                imgTrgPreview=imgTrgPreview,
                imgTrgLabel=imgTrgLabel,
                dicFrameLabelSemSeg=dicFrameLabelSemSeg,
                dicCLT=dicCLT,
            )

            ################################################################################
            dicResSkel = self._SkeletonizeSemSeg(
                imgTrgPreview=imgTrgPreview,
                imgTrgLabel=imgTrgLabel,
                dicFrameLabelSemSeg=dicFrameLabelSemSeg,
                dicCLT=dicCLT,
            )

            if dicResSkel["bOK"] is False:
                print("ERROR: {0}".format(dicResSkel["sMsg"]))
            elif dicResSkel["bOK"] is True:
                if dicResSkel["bProcessed"] is False:
                    print("INFO: {0}".format(dicResSkel["sMsg"]))
                # endif
            # endif

            ################################################################################
            # Store processed label data
            sPathImgTrg = self.dicTrgImgType.get("label-data").get("sPathImgTrg")
            sFilenameConfig = self.dicTrgImgType.get("label-data").get("sFileImgTrg")
            dicExport = {
                "sId": "${filebasename}",
                "lTypes": dicFrameLabelSemSeg.get("lFlatTypes"),
            }

            if dicCamView is not None:
                dicExport["mCamera"] = dicCamView

            else:
                dicCam = dicFrameLabelSemSeg.get("mCamera")
                if dicCam is not None:
                    dicExport["mCamera"] = dicCam
                # endif
            # endif

            dicCamera = dicExport.get("mCamera")
            if anybase.config.IsConfigType(dicCamera, "/anycam/cameraview/pin/std:1.0"):
                dicExport["mCameraCV"] = {
                    "sDTI": "/anycam/opencv/pin/std:1.0",
                    "sInfo": "Image origin at top-left, x/y-axis are right/down. Pixel index refers to top-left of pixel. [0,0] refers to the top-left corner of the top-left pixel.",
                    "lFocLenXY_pix": dicCamera["lFocLenXY_pix"],
                    "lImgCtrXY_pix": [
                        dicCamera["lPixCntXY"][0] / 2.0 + dicCamera["lImgCtrXY_pix"][0],
                        dicCamera["lPixCntXY"][1] / 2.0 - dicCamera["lImgCtrXY_pix"][1],
                    ],
                    "lImgCtrXY_pix/doc": "Principal point relative to top-left image corner",
                    "lPixCntXY": dicCamera["lPixCntXY"],
                    "lAxes": [
                        dicCamera["lAxes"][0],
                        [-x for x in dicCamera["lAxes"][1]],
                        [-x for x in dicCamera["lAxes"][2]],
                    ],
                    "lAxes/doc": "Axes of camera in world CS. lAxes[0] pointing RIGHT, lAxes[1] pointing DOWN, lAxes[2] pointing into the scene",
                    "lOrig_m": dicCamera["lOrig_m"],
                }

            elif anybase.config.IsConfigType(dicCamera, "/anycam/cameraview/pano/equidist:1.0"):
                dicExport["mCameraCV"] = {
                    "sDTI": "/anycam/opencv/pano/equidist:1.0",
                    "sInfo": "Image origin at top-left, x/y-axis are right/down. Pixel index refers to top-left of pixel. [0,0] refers to the top-left corner of the top-left pixel.",
                    "lPixCntXY": dicCamera["lPixCntXY"],
                    "lImgCtrXY_pix": [
                        dicCamera["lPixCntXY"][0] / 2.0,
                        dicCamera["lPixCntXY"][1] / 2.0,
                    ],
                    "lFovXY_deg": dicCamera["lFovXY_deg"],
                    "lAxes": [
                        dicCamera["lAxes"][0],
                        [-x for x in dicCamera["lAxes"][1]],
                        [-x for x in dicCamera["lAxes"][2]],
                    ],
                    "lAxes/doc": "Axes of camera in world CS. lAxes[0] pointing RIGHT, lAxes[1] pointing DOWN, lAxes[2] pointing into the scene",
                    "lOrig_m": dicCamera["lOrig_m"],
                }
            # endif

            cathcfg.Save(
                (sPathImgTrg, sFilenameConfig),
                dicExport,
                sDTI="/anytruth/render/labeltypes/semseg:1.0",
            )

            sFpImgTrg = self.dicTrgImgType.get("label-image").get("sFpImgTrg")
            cv2.imwrite(sFpImgTrg, imgTrgLabel)

            sFpImgTrg = self.dicTrgImgType.get("preview").get("sFpImgTrg")
            cv2.imwrite(sFpImgTrg, imgTrgPreview)

            iTrgFrame += self.iFrameStep
            iTrgFrameIdx += 1
        # endwhile iTrgFrame

    # enddef

    ################################################################################
    def _ConstructRegExFromWildcards(self, _lWildcards):
        # Construct regular expressions from box types
        lRegEx = []
        for sWildcard in _lWildcards:
            if len(sWildcard) == 0:
                continue
            # endif
            sWildcard = sWildcard.replace(".", r"\.")
            if sWildcard[-1] == "*":
                sWildcard = sWildcard[0:-1].replace("*", r"[\w]*") + r"[\w\.]*"
            else:
                sWildcard = sWildcard.replace("*", r"[\w]*")
            # endif
            sWildcard = sWildcard.replace("?", ".")
            lRegEx.append(re.compile(sWildcard))
        # endfor

        return lRegEx

    # enddef

    ###################################################################################
    def _ReadLabelConfig(self, _iTrgFrame):
        from anybase import config

        # Read label config file
        dicLabelTypes = config.Load(
            (self.pathSrcMain, "Frame_{0:04d}.json".format(_iTrgFrame)),
            sDTI="/anytruth/render/labeltypes/raw:1.0",
        )
        lLabelTypes = dicLabelTypes.get("lTypes")
        iLabelTypeCount = len(lLabelTypes)
        iColorNormValue = dicLabelTypes.get("iColorNormValue")
        iTotalNormValue = iColorNormValue * iLabelTypeCount + 1

        # Create flat list of all label types including shader label types
        iMaxInstCnt = 0
        lFlatTypes = [{"sId": "Undefined", "iIdx": 0, "lColor": [0, 0, 0], "iInstanceCount": 1}]

        dicTypeIdx = {"Undefined": 0}

        iIdx = 1
        for dicType in lLabelTypes:
            iMaxInstCnt = max(iMaxInstCnt, dicType.get("iInstanceCount"))
            lFlatTypes.append(
                {
                    "sId": dicType.get("sId"),
                    "iIdx": iIdx,
                    "lColor": dicType.get("lColor"),
                    "iInstanceCount": dicType.get("iInstanceCount"),
                    "iSubInstCount": 0,
                    "mInstances": dicType.get("mInstances"),
                }
            )
            dicTypeIdx[dicType.get("sId")] = iIdx
            iIdx += 1

            iShMaxInstCnt = dicType.get("iShaderMaxInstCnt")
            iMaxInstCnt = max(iMaxInstCnt, iShMaxInstCnt)
            lShTyp = dicType.get("lShaderTypes")
            for dicShTyp in lShTyp:
                sShId = dicShTyp.get("sId")
                if sShId in dicTypeIdx:
                    continue
                # endif

                lColor = dicShTyp.get("lColor")
                if lColor is None:
                    lColor = dicType.get("lColor")
                # endif

                lFlatTypes.append(
                    {
                        "sId": sShId,
                        "iIdx": iIdx,
                        "lColor": lColor,
                        "iInstanceCount": dicType.get("iInstanceCount"),
                        "iSubInstCount": iShMaxInstCnt,
                    }
                )
                dicTypeIdx[dicShTyp.get("sId")] = iIdx
                iIdx += 1
            # endfor
        # endfor

        # distribute vertex group data to types
        for dicType in lLabelTypes:
            dicInstances = dicType.get("mInstances")
            for sInst in dicInstances:
                dicInst = dicInstances.get(sInst)
                dicVexGrpTyps = dicInst.get("mVertexGroups")
                if dicVexGrpTyps is None:
                    continue
                # endif
                for sVgLabelType in dicVexGrpTyps:
                    dicVexGrp = dicVexGrpTyps.get(sVgLabelType)
                    dicFlatType = next((x for x in lFlatTypes if x.get("sId") == sVgLabelType), None)
                    if dicFlatType is None:
                        dicFlatType = {
                            "sId": sVgLabelType,
                            "iIdx": iIdx,
                            "lColor": [],
                            "iInstanceCount": dicType.get("iInstanceCount"),
                            "iSubInstCount": 0,
                        }
                        lFlatTypes.append(dicFlatType)
                        iIdx += 1
                    # endif
                    dicFlatVexGrp = dicFlatType.get("mVertexGroups")
                    if dicFlatVexGrp is None:
                        dicFlatVexGrp = dicFlatType["mVertexGroups"] = {}
                    # endif

                    dicFlatVexGrpInst = dicFlatVexGrp.get(sInst)
                    if dicFlatVexGrpInst is None:
                        dicFlatVexGrpInst = dicFlatVexGrp[sInst] = {}
                    # endif
                    dicFlatVexGrpInst.update(dicVexGrp)
                # endfor vertex group label types
            # endfor instances
        # endfor label types

        # Select output image bit depth
        if iMaxInstCnt > 254 or len(lFlatTypes) > 254:
            xTrgType = np.uint16
        else:
            xTrgType = np.uint8
        # endif
        # xTrgTypeInfo = np.iinfo(xTrgType)

        # Write label types json
        dicLabelSemSeg = {
            "sId": "${filebasename}",
            "iTotalNormValue": iTotalNormValue,
            "xTrgType": xTrgType,
            "mCamera": dicLabelTypes.get("mCamera"),
            "lLabelTypes": lLabelTypes,
            "lFlatTypes": lFlatTypes,
            "mTypeIdx": dicTypeIdx,
        }
        # config.Save(self.sPathTrgMain, "AT_Label_SemSeg.json", dicLabelSemSeg, sDTI="/anytruth/render/labeltypes/semseg:1.0")

        return dicLabelSemSeg

    # endif

    ################################################################################
    def _GetSkeletonBones(self, _dicSkel):
        dicBones = {}
        for sBone in _dicSkel:
            dicChildBones = _dicSkel.get(sBone)
            dicChildren = self._GetSkeletonBones(dicChildBones)
            dicBones[sBone] = {"lChildren": [sKey for sKey in dicChildBones]}
            for sChild in dicChildBones:
                dicChildren[sChild]["sParent"] = sBone
            # endfor
            dicBones.update(dicChildren)
        # endfor

        return dicBones

    # enddef

    ################################################################################
    # Pre-process skeleton types for easier processing later on
    def _ProcessSkeletonTypes(self, _dicSkelTypes):
        if _dicSkelTypes is None:
            return {}
        # endif

        dicFlatSkelTypes = {}
        for sSkelType in _dicSkelTypes:
            dicSkelType = _dicSkelTypes.get(sSkelType)
            dicFlatSkelTypes[sSkelType] = self._GetSkeletonBones(dicSkelType)
        # endfor

        return dicFlatSkelTypes

    # enddef

    ###################################################################
    # Create frame name from frame index
    def _GetFrameName(self, iFrame):
        return "Frame_{0:04d}".format(iFrame)

    # enddef

    ###################################################################
    # Convert raw label image to SemSeg label id image and preview image
    def _ProcessImgRawSemSeg(self, *, iTrgFrame, dicFrameLabelSemSeg):
        xTrgType = dicFrameLabelSemSeg.get("xTrgType")
        xTrgTypeInfo = np.iinfo(xTrgType)
        iTotalNormValue = dicFrameLabelSemSeg.get("iTotalNormValue")
        lLabelTypes = dicFrameLabelSemSeg.get("lLabelTypes")
        iLabelTypeCount = len(lLabelTypes)
        dicTypeIdx = dicFrameLabelSemSeg.get("mTypeIdx")

        ###################################################################
        # Check whether raw label image file exists
        sFrameName = self._GetFrameName(iTrgFrame)
        sFileImgSrc = "{0}.exr".format(sFrameName)

        sFpImgSrc = os.path.normpath(os.path.join(self.pathSrcMain, sFileImgSrc))
        if not os.path.exists(sFpImgSrc):
            return {
                "bOK": False,
                "sMsg": "Frame '{0}' does not exist. Skipping...".format(sFileImgSrc),
            }
        # endif

        # Load source image
        # print("Loading image: {0}".format(sFpImgSrc))
        imgSrcImg = cv2.imread(sFpImgSrc, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if imgSrcImg is None:
            print("Error loading image: {0}".format(sFpImgSrc))
            return None
        # endif
        iSrcRows, iSrcCols, iSrcChnl = imgSrcImg.shape

        # Create output arrays
        imgTrgL = np.zeros((iSrcRows, iSrcCols, 3), dtype=xTrgType)
        imgTrgP = np.zeros((iSrcRows, iSrcCols, 3), dtype=np.uint8)

        # typSrcImg = imgSrcImg.dtype
        # fSrcMin = np.amin(imgSrcImg)
        # fSrcMax = np.amax(imgSrcImg)
        imgR = imgSrcImg.astype(np.float64)

        # Label Type in red channel
        imgMask = imgR[:, :, self.iChNorm] > 1e-4
        # fMin = imgR.min()
        # fMax = imgR.max()

        imgLIT = np.divide(imgR[:, :, self.iChType], imgR[:, :, self.iChNorm], where=imgMask)
        imgLIT = np.around(imgLIT * iTotalNormValue)
        # imgLitDebug  = imgLIT.copy()
        imgLIT = np.subtract(imgLIT, 1, where=imgMask)
        imgType = np.mod(imgLIT, iLabelTypeCount, where=imgMask).astype(xTrgType)
        imgInst = np.floor_divide(imgLIT, iLabelTypeCount, where=imgMask).astype(xTrgType)

        # Shader label instance type
        imgSIT = np.divide(imgR[:, :, self.iChShType], imgR[:, :, self.iChNorm], where=imgMask)

        # !!!DEBUG
        # sPathDebug = r"[debug path]"
        # imgDbgBool = np.zeros((iSrcRows, iSrcCols), dtype=np.uint8)

        for iIdx, dicType in enumerate(lLabelTypes):
            sTypeId = dicType.get("sId")
            iFlatIdx = dicTypeIdx.get(sTypeId)

            # if dicType.get("sId") != "Sign":
            # 	continue
            # # endif

            lColor = np.array(
                [x * (xTrgTypeInfo.max - xTrgTypeInfo.min) + xTrgTypeInfo.min for x in dicType.get("lColor")]
            ).astype(xTrgType)

            imgMaskType = np.logical_and(imgType == iIdx, imgMask)
            aIdxType = np.argwhere(imgMaskType)

            imgTrgL[aIdxType[:, 0], aIdxType[:, 1], self.iTrgChType] = iFlatIdx
            imgTrgL[aIdxType[:, 0], aIdxType[:, 1], self.iTrgChInst] = imgInst[aIdxType[:, 0], aIdxType[:, 1]]
            imgTrgP[aIdxType[:, 0], aIdxType[:, 1]] = np.flip(lColor)

            lShType = dicType.get("lShaderTypes")
            iShLabelTypeCnt = len(lShType)
            if iShLabelTypeCnt > 0:
                iShMaxInstCnt = dicType.get("iShaderMaxInstCnt")
                iShNormValue = iShMaxInstCnt * iShLabelTypeCnt + 1

                # imgDbgBool[imgMaskType] = 255
                # cv2.imwrite(os.path.join(sPathDebug, "imgMaskType.png"), imgDbgBool)
                # imgDbgBool[:, :] = 0

                imgShMask = np.logical_and(imgSIT > 1e-4, imgMaskType)
                # imgDbgBool[imgShMask] = 255
                # cv2.imwrite(os.path.join(sPathDebug, "imgShMask.png"), imgDbgBool)
                # imgDbgBool[:, :] = 0

                imgActSIT = np.around(imgSIT * iShNormValue)
                imgActSIT = np.subtract(imgActSIT, 1, where=imgShMask)
                imgShType = np.mod(imgActSIT, iShLabelTypeCnt, where=imgShMask).astype(xTrgType)
                imgShInst = np.floor_divide(imgActSIT, iShLabelTypeCnt, where=imgShMask).astype(xTrgType)

                for iShIdx, dicShType in enumerate(lShType):
                    sShTypeId = dicShType.get("sId")

                    # if sShTypeId != "Skeleton;Std;Right_Eye":
                    #     continue
                    # # endif
                    # Treat skeleton shader types differnt to normal shader types.
                    # For skeleton shader types, the self.iTrgChShInst receives the flat
                    # shader type index and the iTrgChType channel receives the parent type index.
                    #
                    # For standard shader types, the self.iTrgChShInst receives the shader instance
                    # and the iTrgChType channel receives the flat shader type index
                    if sShTypeId.startswith("Skeleton;"):
                        iFlatIdx = dicTypeIdx.get(sTypeId)
                        sShTypeMode = "SKELETON"

                    else:
                        iFlatIdx = dicTypeIdx.get(sShTypeId)
                        sShTypeMode = "DEFAULT"
                    # endif

                    lShColor = dicShType.get("lColor")
                    if lShColor is None:
                        lShColor = dicType.get("lColor")
                    # endif

                    lColor = np.array(
                        [x * (xTrgTypeInfo.max - xTrgTypeInfo.min) + xTrgTypeInfo.min for x in lShColor]
                    ).astype(xTrgType)

                    maskIdxShType = np.logical_and(imgShType == iShIdx, imgShMask)
                    aIdxShType = np.argwhere(maskIdxShType)
                    imgTrgL[aIdxShType[:, 0], aIdxShType[:, 1], self.iTrgChType] = iFlatIdx

                    if sShTypeMode == "SKELETON":
                        # Set flat index of shader type
                        imgTrgL[aIdxShType[:, 0], aIdxShType[:, 1], self.iTrgChShInst] = dicTypeIdx.get(sShTypeId)
                    else:
                        # Set shader type instance index
                        imgTrgL[aIdxShType[:, 0], aIdxShType[:, 1], self.iTrgChShInst] = imgShInst[
                            aIdxShType[:, 0], aIdxShType[:, 1]
                        ]
                    # endif
                    imgTrgP[aIdxShType[:, 0], aIdxShType[:, 1]] = np.flip(lColor)
                # endfor
            # endif "has shader label types"
        # endfor

        return {
            "bOK": True,
            "sMsg": "",
            "imgTrgPreview": imgTrgP,
            "imgTrgLabel": imgTrgL,
        }

    # enddef

    ################################################################################
    def _ProcessSkeletons(self, *, xView, imgTrgPreview, imgTrgLabel, dicFrameLabelSemSeg, dicCLT):
        # Get Skeleton Types
        dicSkel = None
        dicProcess = dicCLT.get("mProcess")
        if dicProcess is not None:
            dicSkel = dicProcess.get("skeleton")
        # endif

        if dicSkel is None:
            # don't process skeletons by default, if none is defined
            print("Not processing skeletons due to missing settings...")
            return
        # endif
        print("Processing skeletons...", flush=True)

        dicSkelTypes = self._ProcessSkeletonTypes(dicSkel.get("mTypes"))
        lPreviewSkelTypes = dicSkel.get("lPreviewTypes", ["*"])
        bDoPreviewSkeletons = dicSkel.get("iDoPreview", 1) != 0
        dicPreview = dicSkel.get("mPreview", {})
        iPreviewJointRadius = dicPreview.get("iJointRadius", 5)
        iPreviewLineWidth = dicPreview.get("iLineWidth", 2)

        iTrgRowCnt, iTrgColCnt, iTrgChCnt = imgTrgLabel.shape

        # Loop over all label types
        for dicFlatType in dicFrameLabelSemSeg.get("lFlatTypes"):
            # sTypeId = dicFlatType.get("sId")
            dicInstances = dicFlatType.get("mInstances")
            if dicInstances is None:
                continue
            # endif

            # Loop over instances
            for sInstIdx in dicInstances:
                # iInstIdx = int(sInstIdx)
                dicInst = dicInstances.get(sInstIdx)

                # Process pose skeletons, if any
                dicPoses3d = dicInst.get("mPoses3d")
                if dicPoses3d is not None:
                    for sPoseId in dicPoses3d:
                        dicPose = dicPoses3d.get(sPoseId)
                        dicBones = dicPose.get("mBones")
                        # Project all bones to 2d
                        for sBone in dicBones:
                            dicBone = dicBones.get(sBone)
                            lPnts3d = [
                                dicBone.get("lHead"),
                                dicBone.get("lTail"),
                            ]

                            if xView is not None:
                                iFlatShTypeIdx = None
                                xMatch = self.reAtBone.match(sBone)
                                if xMatch is not None:
                                    sSkelId = xMatch.group("skel")
                                    sBoneId = xMatch.group("bone")
                                    sShTypeId = "Skeleton;{};{}".format(sSkelId, sBoneId)
                                    iFlatShTypeIdx = dicFrameLabelSemSeg["mTypeIdx"].get(sShTypeId)
                                # endif

                                lPnts2d_pix, lInFront = xView.ProjectToImage(lPnts3d)
                                lIsOccluded = []
                                lIsSelfOcc = []
                                for aPnt2d_pix, bInFront in zip(lPnts2d_pix, lInFront):
                                    iR = int(round(aPnt2d_pix[1]))
                                    iC = int(round(aPnt2d_pix[0]))

                                    if not bInFront or iR < 0 or iR >= iTrgRowCnt or iC < 0 or iC >= iTrgColCnt:
                                        lIsOccluded.append(True)
                                        lIsSelfOcc.append(None)
                                        continue
                                    # endif

                                    iTypeAtPix = int(imgTrgLabel[iR, iC, self.iTrgChType])
                                    iInstAtPix = int(imgTrgLabel[iR, iC, self.iTrgChInst])
                                    iShTypeAtPix = int(imgTrgLabel[iR, iC, self.iTrgChShInst])
                                    bIsOccluded = dicFlatType["iIdx"] != iTypeAtPix or dicInst["iIdx"] != iInstAtPix
                                    if iFlatShTypeIdx is None:
                                        bIsSelfOcc = None
                                    else:
                                        bIsSelfOcc = (not bIsOccluded) and (iFlatShTypeIdx != iShTypeAtPix)
                                    # endif

                                    lIsOccluded.append(bIsOccluded)
                                    lIsSelfOcc.append(bIsSelfOcc)
                                # endfor

                                dicBone["mImage"] = {
                                    "bHeadOccluded": lIsOccluded[0],
                                    "bTailOccluded": lIsOccluded[1],
                                    "bHeadSelfOcc": lIsSelfOcc[0],
                                    "bTailSelfOcc": lIsSelfOcc[1],
                                    "lHead": lPnts2d_pix[0] if lInFront[0] else None,
                                    "lTail": lPnts2d_pix[1] if lInFront[1] else None,
                                }
                            # endif
                        # endfor bones

                        dicPoseSkelTypes = dicPose.get("mTypes")
                        if dicPoseSkelTypes is None:
                            dicPoseSkelTypes = dicPose["mTypes"] = {}
                        # endif

                        # Process specified skeleton types
                        for sSkelType in dicSkelTypes:
                            bSkelOK = True
                            dicSTBones = copy.deepcopy(dicSkelTypes.get(sSkelType))
                            for sSTBone in dicSTBones:
                                dicSTBone = dicSTBones.get(sSTBone)
                                sBoneName = "AT.Label;{0};{1}".format(sSkelType, sSTBone)
                                dicBone = dicBones.get(sBoneName)
                                if dicBone is None:
                                    bSkelOK = False
                                    print(f"WARNING: Bone '{sBoneName}' not found for skeleton type '{sSkelType}'")
                                    dicSTBone["sWARNING"] = f"Bone '{sBoneName}' not found"
                                    continue
                                # endif
                                dicSTBone["lHead"] = dicBone.get("lHead")
                                dicSTBone["lTail"] = dicBone.get("lTail")
                                dicSTBone["lAxes"] = dicBone.get("lAxes")
                                dicSTBone["mImage"] = dicBone.get("mImage")
                            # endfor

                            # if bSkelOK:
                            dicPoseSkelTypes[sSkelType] = dicSTBones
                            # # endif
                        # endfor skeleton types

                        # Draw skeleton in preview image
                        if bDoPreviewSkeletons and xView is not None:
                            lPreviewBones = []
                            if len(lPreviewSkelTypes) == 0:
                                lPreviewBones = [dicBones]
                            else:
                                for sSkelType in lPreviewSkelTypes:
                                    dicSTBones = dicPose.get("mTypes").get(sSkelType)
                                    if dicSTBones is not None:
                                        lPreviewBones.append(dicSTBones)
                                    # endif
                                # endfor
                            # endif

                            tColorJoint = (50, 255, 50)
                            tColorBone = (255, 50, 50)
                            tColorBoneOccluded = (255, 240, 240)

                            for dicBones in lPreviewBones:
                                for sBone in dicBones:
                                    dicBone = dicBones.get(sBone)
                                    dicBoneImage = dicBone.get("mImage")
                                    if dicBoneImage is None:
                                        print(
                                            f"WARNING: Bone '{sBone}' has no projection to the image. No preview will be generated."
                                        )
                                        continue
                                    # endif

                                    lHead = dicBoneImage.get("lHead")
                                    if lHead is None:
                                        continue
                                    # endif
                                    bHeadOccluded = dicBoneImage["bHeadOccluded"]
                                    bHeadSelfOcc = dicBoneImage["bHeadSelfOcc"]

                                    iThickness = -1
                                    if bHeadOccluded is True or bHeadSelfOcc is True:
                                        iThickness = 1
                                    # endif

                                    tHead = tuple((int(round(x)) for x in lHead))
                                    cv2.circle(
                                        imgTrgPreview,
                                        tHead,
                                        iPreviewJointRadius,
                                        tColorJoint,
                                        iThickness,
                                    )
                                    if bHeadSelfOcc is True:
                                        cv2.circle(
                                            imgTrgPreview,
                                            tHead,
                                            2 * iPreviewJointRadius,
                                            tColorJoint,
                                            iThickness,
                                        )
                                    # endif

                                    for sChild in dicBone.get("lChildren"):
                                        dicChild = dicBones.get(sChild)
                                        dicChildImage = dicChild.get("mImage")
                                        if dicChildImage is None:
                                            print(
                                                f"WARNING: Child-bone '{sChild}' of bone '{sBone}' has no projection to the image. No preview will be generated."
                                            )
                                            continue
                                        # endif

                                        lTail = dicChildImage.get("lHead")
                                        if lTail is None:
                                            continue
                                        # endif
                                        bTailOccluded = dicChildImage["bHeadOccluded"]
                                        bTailSelfOcc = dicChildImage["bHeadSelfOcc"]
                                        bBoneOccluded = (
                                            bTailOccluded is True
                                            or bHeadOccluded is True
                                            or bTailSelfOcc is True
                                            or bHeadSelfOcc is True
                                        )
                                        tColorLine = tColorBoneOccluded if bBoneOccluded else tColorBone

                                        tTail = tuple((int(round(x)) for x in lTail))
                                        cv2.line(
                                            imgTrgPreview,
                                            tHead,
                                            tTail,
                                            tColorLine,
                                            iPreviewLineWidth,
                                        )
                                    # endfor bone children
                                # endfor bones
                            # endfor preview bone collections
                        # endif do preview skeletons
                    # endfor poses
                # endif has 3d poses
            # endfor instances
        # endfor label types

    # enddef

    ################################################################################
    # Test wether given types matches a given list of regexs
    def _IsOfType(self, _sType, _lRegEx):
        xMatch = None
        for reType in _lRegEx:
            xMatch = reType.match(_sType)
            if xMatch is not None:
                break
            # endif
        # endfor
        return xMatch is not None

    # endif

    ################################################################################
    # Process 3d
    def _ProcessIdealInstanceBoxes2d(self, *, imgTrgPreview, dicFrameLabelSemSeg, dicCLT):
        print("Processing ideal 2d boxes...", flush=True)

        dicIdealBox2d: dict = None
        dicProcess = dicCLT.get("mProcess")
        if dicProcess is not None:
            dicIdealBox2d = dicProcess.get("ideal_box2d")
        # endif

        # Get Box-3d projection settings
        if dicIdealBox2d is None:
            # project all 3d boxes by default
            print("Using default 2d-box settings...")
            # lReBox2dEvalTypes = self._ConstructRegExFromWildcards(["*"])
            lReBox2dPreviewTypes = self._ConstructRegExFromWildcards(["*"])
            bDoBox2dPreview = True
            dicBox2dPreview = {}
        else:
            # lReBox2dEvalTypes = self._ConstructRegExFromWildcards(dicIdealBox2d.get("lEvalTypes", ["*"]))
            lReBox2dPreviewTypes = self._ConstructRegExFromWildcards(dicIdealBox2d.get("lPreviewTypes", ["*"]))
            bDoBox2dPreview = dicIdealBox2d.get("iDoPreview")
            dicBox2dPreview = dicIdealBox2d.get("mPreview", {})
        # endif

        iBox2dPreviewLineWidth = dicBox2dPreview.get("iLineWidth", 2)

        # Loop over all label types
        for dicFlatType in dicFrameLabelSemSeg.get("lFlatTypes"):
            sTypeId = dicFlatType.get("sId")
            dicInstances = dicFlatType.get("mInstances")
            if dicInstances is None:
                continue
            # endif

            # Loop over instances
            for sInstIdx in dicInstances:
                # iInstIdx = int(sInstIdx)
                dicInst = dicInstances.get(sInstIdx)

                # Process 2d Boxes, if any
                dicBox2d = dicInst.get("mBox2d")
                if dicBox2d is not None and self._IsOfType(sTypeId, lReBox2dPreviewTypes) and bDoBox2dPreview:
                    lMinXY = dicBox2d["lMinXY"]
                    lMaxXY = dicBox2d["lMaxXY"]
                    cv2.rectangle(
                        imgTrgPreview,
                        (int(round(lMinXY[0])), int(round(lMinXY[1]))),
                        (int(round(lMaxXY[0])), int(round(lMaxXY[1]))),
                        (0, 0, 255),
                        iBox2dPreviewLineWidth,
                    )

                    # lPnts2d_pix = [
                    #     [lMinXY[0], lMinXY[1]],
                    #     [lMaxXY[0], lMinXY[1]],
                    #     [lMaxXY[0], lMaxXY[1]],
                    #     [lMinXY[0], lMaxXY[1]],
                    #     [lMinXY[0], lMinXY[1]],
                    # ]
                    # lDrawPnts = np.array(lPnts2d_pix, dtype=np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(
                    #     imgTrgPreview,
                    #     [lDrawPnts],
                    #     True,
                    #     (0, 255, 255),
                    #     iBox2dPreviewLineWidth,
                    # )
                # endif
            # endfor instance
        # endfor type

    # enddef

    ################################################################################
    # Project 3d label data to image, if projection for active camera is supported
    def _ProjectLabelDataToImage(self, *, xView, imgTrgPreview, dicFrameLabelSemSeg, dicCLT):
        print("Projecting 3d label data to image...", flush=True)
        if xView is None:
            print(">> Active camera not supported for projection")
            return
        # endif

        dicPrjBox3d = None
        dicPrjVexGrp = None
        dicProcess = dicCLT.get("mProcess")
        if dicProcess is not None:
            dicPrjBox3d = dicProcess.get("project_box3d")
            dicPrjVexGrp = dicProcess.get("project_vertex_groups")
        # endif

        # Get Box-3d projection settings
        if dicPrjBox3d is None:
            # project all 3d boxes by default
            print("Using default box3d projection settings...")
            lReBox3dEvalTypes = self._ConstructRegExFromWildcards(["*"])
            lReBox3dPreviewTypes = self._ConstructRegExFromWildcards(["*"])
            bDoBox3dPreview = True
            dicBox3dPreview = {}
        else:
            lReBox3dEvalTypes = self._ConstructRegExFromWildcards(dicPrjBox3d.get("lEvalTypes", ["*"]))
            lReBox3dPreviewTypes = self._ConstructRegExFromWildcards(dicPrjBox3d.get("lPreviewTypes", ["*"]))
            bDoBox3dPreview = dicPrjBox3d.get("iDoPreview")
            dicBox3dPreview = dicPrjBox3d.get("mPreview", {})
        # endif

        # Get Vertex Group projection settings
        if dicPrjVexGrp is None:
            # project all 3d boxes by default
            print("Using default vertex group projection settings...")
            lReVexGrpEvalTypes = self._ConstructRegExFromWildcards(["*"])
            lReVexGrpPreviewTypes = self._ConstructRegExFromWildcards(["*"])
            bDoVexGrpPreview = True
            dicVexGrpPreview = {}
        else:
            lReVexGrpEvalTypes = self._ConstructRegExFromWildcards(dicPrjVexGrp.get("lEvalTypes", ["*"]))
            lReVexGrpPreviewTypes = self._ConstructRegExFromWildcards(dicPrjVexGrp.get("lPreviewTypes", ["*"]))
            bDoVexGrpPreview = dicPrjVexGrp.get("iDoPreview")
            dicVexGrpPreview = dicPrjVexGrp.get("mPreview", {})
        # endif

        iBox3dPreviewLineWidth = dicBox3dPreview.get("iLineWidth", 2)
        iVexGrpPreviewLineWidth = dicVexGrpPreview.get("iLineWidth", 1)

        # Loop over all label types
        for dicFlatType in dicFrameLabelSemSeg.get("lFlatTypes"):
            sTypeId = dicFlatType.get("sId")
            dicInstances = dicFlatType.get("mInstances")
            if dicInstances is None:
                continue
            # endif

            # Loop over instances
            for sInstIdx in dicInstances:
                # iInstIdx = int(sInstIdx)
                dicInst = dicInstances.get(sInstIdx)

                # Process 3d Boxes, if any
                dicBox3d = dicInst.get("mBox3d")
                if dicBox3d is not None and self._IsOfType(sTypeId, lReBox3dEvalTypes):
                    aCtr = np.array(dicBox3d.get("lCenter"))
                    aSize = np.array(dicBox3d.get("lSize"))
                    aAxes = np.array(dicBox3d.get("lAxes"))
                    aCamAxes = xView.DirsToCameraFrame(aAxes)

                    # Create vertices for all sides of box
                    aSizeH = aSize / 2.0
                    dicSides = {}
                    lCtr2d_pix, lInFront = xView.ProjectToImage(np.array([dicBox3d.get("lCenter")]))
                    if lInFront[0] is True:
                        dicSides["center-point"] = lCtr2d_pix
                    # endif

                    aPnts = np.empty((4, 3))
                    lSides = ["front", "rear", "left", "right", "top", "bottom"]
                    # lSideTextOrigIdx = [1, 1, 2, 2, 0, 2]
                    iSideIdx = 0
                    for iIdx in range(3):
                        iIdxX = (iIdx + 1) % 3
                        iIdxY = (iIdx + 2) % 3

                        for fSign in [1.0, -1.0]:
                            aCamNorm = fSign * aCamAxes[iIdx]
                            aOffset = aCtr + fSign * aAxes[iIdx] * aSizeH[iIdx]

                            aX = aSizeH[iIdxX] * aAxes[iIdxX]
                            aY = aSizeH[iIdxY] * aAxes[iIdxY]
                            aPnts[0] = aOffset + aX + aY
                            aPnts[1] = aOffset - aX + aY
                            aPnts[2] = aOffset - aX - aY
                            aPnts[3] = aOffset + aX - aY

                            aCamPnts = xView.PointsToCameraFrame(aPnts)
                            # Test whether box plane is visible
                            if np.all(np.dot(aCamPnts, aCamNorm) < 0):
                                lPnts2d_pix, lInFront = xView.ProjectToImage(aPnts)
                                if all(lInFront):
                                    sSide = lSides[iSideIdx]
                                    dicSides[sSide] = lPnts2d_pix
                                    lDrawPnts = np.array(lPnts2d_pix, dtype=np.int32).reshape((-1, 1, 2))

                                    # Check whether projection should be drawn in preview image
                                    if bDoBox3dPreview and self._IsOfType(sTypeId, lReBox3dPreviewTypes):
                                        cv2.polylines(
                                            imgTrgPreview,
                                            [lDrawPnts],
                                            True,
                                            (0, 255, 255),
                                            iBox3dPreviewLineWidth,
                                        )

                                        # Using cv2.putText() method
                                        # aPnts2d_pix = np.array([(e[0], e[1])
                                        #                   for e in lPnts2d_pix], dtype=[('x', 'f4'), ('y', 'f4')])
                                        # aSortPnts = np.sort(aPnts2d_pix, order=('x', 'y'))
                                        # aSortPnts = np.append([aSortPnts["x"]], [aSortPnts["y"]], axis=0).transpose()
                                        # iOrigIdx = lSideTextOrigIdx[iSideIdx]
                                        # lPnts = aSortPnts[iOrigIdx].round().tolist()
                                        # tPos = tuple((int(x) for x in lPnts))
                                        # imgTrgPreview = cv2.putText(imgTrgP, sSide, tPos, cv2.FONT_HERSHEY_SIMPLEX,
                                        # 				1, (0, 255, 255), 2, cv2.LINE_AA)
                                    # endif
                                # endif
                            # endif
                            iSideIdx += 1
                        # endfor sign
                    # endfor axes
                    dicBox3d["mImage"] = dicSides
                # endif has 3d box

                # Process vertex groups, if any
                dicVexGroups = dicInst.get("mVertexGroups")
                if dicVexGroups is not None and self._IsOfType(sTypeId, lReVexGrpEvalTypes):
                    for sVgLabelType in dicVexGroups:
                        dicVgType = dicVexGroups.get(sVgLabelType)
                        for sVgInst in dicVgType:
                            dicVexGrpIds = dicVgType.get(sVgInst)
                            for sVgId in dicVexGrpIds:
                                lVexLists = dicVexGrpIds.get(sVgId)
                                for dicVexList in lVexLists:
                                    sType = dicVexList.get("sType")
                                    if sType == "LINESTRIP":
                                        lPnts2d_pix, lInFront = xView.ProjectToImage(dicVexList.get("lVex"))
                                        lDrawPnts2d_pix = [x for i, x in enumerate(lPnts2d_pix) if lInFront[i]]
                                        lPnts2d_pix = [x if lInFront[i] else None for i, x in enumerate(lPnts2d_pix)]
                                        dicVexList["lImageVex"] = lPnts2d_pix
                                        lDrawPnts = np.array(lDrawPnts2d_pix, dtype=np.int32).reshape((-1, 1, 2))
                                        if bDoVexGrpPreview and self._IsOfType(sTypeId, lReVexGrpPreviewTypes):
                                            cv2.polylines(
                                                imgTrgPreview,
                                                [lDrawPnts],
                                                False,
                                                (255, 0, 255),
                                                iVexGrpPreviewLineWidth,
                                            )
                                        # endif
                                    # endif
                                # endfor
                            # endfor
                        # endfor
                    # endfor
                # endif has vertex groups
            # endfor instances
        # endfor label types

    # enddef

    ################################################################################
    # Find 2d-boxes for instances per type
    def _FindSemSegInstanceBoxes2d(self, *, imgTrgPreview, imgTrgLabel, dicFrameLabelSemSeg, dicCLT):
        dicBox2d = None
        dicProcess = dicCLT.get("mProcess")
        if dicProcess is not None:
            dicBox2d = dicProcess.get("box2d")
        # endif

        if dicBox2d is None:
            print("Using default box2d settings...")
            lReEvalTypes = self._ConstructRegExFromWildcards(["*"])
            lRePreviewTypes = self._ConstructRegExFromWildcards(["*"])
            bDoPreview = True
            dicPreview = {}
        else:
            lReEvalTypes = self._ConstructRegExFromWildcards(dicBox2d.get("lEvalTypes", ["*"]))
            lRePreviewTypes = self._ConstructRegExFromWildcards(dicBox2d.get("lPreviewTypes", ["*"]))
            bDoPreview = dicBox2d.get("iDoPreview", 1) != 0
            dicPreview = dicBox2d.get("mPreview", {})
        # endif

        iPreviewLineWidth = dicPreview.get("iLineWidth", 2)

        print("Finding 2d-boxes...", flush=True)
        for dicFlatType in dicFrameLabelSemSeg.get("lFlatTypes"):
            sTypeId = dicFlatType.get("sId")
            if not self._IsOfType(sTypeId, lReEvalTypes):
                continue
            # endif

            bDoDrawPreview = bDoPreview and self._IsOfType(sTypeId, lRePreviewTypes)

            print("2d-boxes: Eval boxes for type '{0}'".format(sTypeId), flush=True)

            iTypeIdx = dicFlatType.get("iIdx")
            if iTypeIdx == 0:
                continue
            # endif

            lBox = []

            iInstCnt = dicFlatType.get("iInstanceCount")
            iSubInstCnt = dicFlatType.get("iSubInstCount")

            maskType = imgTrgLabel[:, :, self.iTrgChType] == iTypeIdx
            for iInstIdx in range(1, iInstCnt + 1):
                maskInst = imgTrgLabel[:, :, self.iTrgChInst] == iInstIdx

                if iSubInstCnt == 0:
                    aIdxInst = np.argwhere(np.logical_and(maskType, maskInst))
                    if len(aIdxInst) > 0:
                        iMinRow = int(aIdxInst[:, 0].min())
                        iMaxRow = int(aIdxInst[:, 0].max())
                        iMinCol = int(aIdxInst[:, 1].min())
                        iMaxCol = int(aIdxInst[:, 1].max())
                        lBox.append(
                            {
                                "iInstIdx": iInstIdx,
                                "iSubInstIdx": 0,
                                "lRowRange": [iMinRow, iMaxRow],
                                "lColRange": [iMinCol, iMaxCol],
                            }
                        )

                        if bDoDrawPreview:
                            cv2.rectangle(
                                imgTrgPreview,
                                (iMinCol, iMinRow),
                                (iMaxCol, iMaxRow),
                                (0, 0, 255),
                            )
                        # endif
                    # endif
                else:
                    for iSubInstIdx in range(iSubInstCnt):
                        maskSubInst = imgTrgLabel[:, :, self.iTrgChShInst] == iSubInstIdx
                        aIdxInst = np.argwhere(np.logical_and(maskType, np.logical_and(maskInst, maskSubInst)))
                        if len(aIdxInst) > 0:
                            iMinRow = int(aIdxInst[:, 0].min())
                            iMaxRow = int(aIdxInst[:, 0].max())
                            iMinCol = int(aIdxInst[:, 1].min())
                            iMaxCol = int(aIdxInst[:, 1].max())
                            lBox.append(
                                {
                                    "iInstIdx": iInstIdx,
                                    "iSubInstIdx": iSubInstIdx,
                                    "lRowRange": [iMinRow, iMaxRow],
                                    "lColRange": [iMinCol, iMaxCol],
                                }
                            )
                            if bDoDrawPreview:
                                cv2.rectangle(
                                    imgTrgPreview,
                                    (iMinCol, iMinRow),
                                    (iMaxCol, iMaxRow),
                                    (0, 0, 255),
                                    iPreviewLineWidth,
                                )
                            # endif
                        # endif
                    # endfor sub-instance
                # endif
            # endfor instance

            dicFlatType["lBoxes2D"] = lBox
        # endfor FlatType
        print("2d-boxes: fininshed.", flush=True)
        print("", flush=True)

    # enddef

    ################################################################################
    # Skeletonize SemSeg areas for selected types (slow process)
    def _SkeletonizeSemSeg(self, *, imgTrgPreview, imgTrgLabel, dicFrameLabelSemSeg, dicCLT):
        # sPathImgDbg = self.dicTrgImgType.get("debug").get("sPathImgTrg")

        dicSkel = None
        dicProcess = dicCLT.get("mProcess")
        if dicProcess is not None:
            dicSkel = dicProcess.get("skeletonize")
        # endif

        if dicSkel is None:
            # Default behaviour is not to skeletonize anything
            return {
                "bOK": True,
                "bProcessed": False,
                "sMsg": "No skeletonization by default",
            }
        # endif

        print("Skeletonize selected types...", flush=True)

        lReEvalTypes = self._ConstructRegExFromWildcards(dicSkel.get("lEvalTypes", []))
        lRePreviewTypes = self._ConstructRegExFromWildcards(dicSkel.get("lPreviewTypes", ["*"]))
        bDoPreview = dicSkel.get("iDoPreview", 1) != 0

        # if no SemSeg types should be skeletonized, then return directly
        if len(lReEvalTypes) == 0:
            return {
                "bOK": True,
                "bProcessed": False,
                "sMsg": "No type selected for skeletonization",
            }
        # endif

        iSrcRows, iSrcCols, iSrcChnl = imgTrgLabel.shape
        imgSkel = np.zeros((iSrcRows, iSrcCols), dtype=np.float64)

        for dicFlatType in dicFrameLabelSemSeg.get("lFlatTypes"):
            sTypeId = dicFlatType.get("sId")
            if not self._IsOfType(sTypeId, lReEvalTypes):
                continue
            # endif

            print("Skeletonize: Processing type '{0}'".format(sTypeId), flush=True)
            iTypeIdx = dicFlatType.get("iIdx")
            if iTypeIdx == 0:
                continue
            # endif

            bDoDrawPreview = bDoPreview and self._IsOfType(sTypeId, lRePreviewTypes)
            lSkelLines = []

            iInstCnt = dicFlatType.get("iInstanceCount")
            iSubInstCnt = dicFlatType.get("iSubInstCount")

            maskType = imgTrgLabel[:, :, self.iTrgChType] == iTypeIdx
            for iInstIdx in range(1, iInstCnt + 1):
                maskInst = imgTrgLabel[:, :, self.iTrgChInst] == iInstIdx

                if iSubInstCnt == 0:
                    # sFpDebug = os.path.join(sPathImgDbg, "Mask_{0}_{1}_{2}_{3:04d}.png"
                    #                                      .format(sTypeId, iInstIdx, 0, iTrgFrame))

                    aIdxInst = np.argwhere(np.logical_and(maskType, maskInst))
                    if len(aIdxInst) > 0:
                        imgSkel.fill(0)
                        imgSkel[aIdxInst[:, 0], aIdxInst[:, 1]] = 1
                        # cv2.imwrite(sFpDebug, imgSkel)
                        dicSkelLines = self.SkelImage(imgSkel)
                        dicSkelLines["iInstIdx"] = iInstIdx
                        dicSkelLines["iSubInstIdx"] = 0
                        lSkelLines.append(dicSkelLines)
                    # endif
                else:
                    for iSubInstIdx in range(iSubInstCnt):
                        # sFpDebug = os.path.join(sPathImgDbg, "Mask_{0}_{1}_{2}_{3:04d}.png"
                        #                                      .format(sTypeId, iInstIdx,
                        #                                              iSubInstIdx, iTrgFrame))

                        maskSubInst = imgTrgLabel[:, :, self.iTrgChShInst] == iSubInstIdx
                        aIdxInst = np.argwhere(np.logical_and(maskType, np.logical_and(maskInst, maskSubInst)))
                        if len(aIdxInst) > 0:
                            imgSkel.fill(0)
                            imgSkel[aIdxInst[:, 0], aIdxInst[:, 1]] = 1
                            # cv2.imwrite(sFpDebug, imgSkel)
                            dicSkelLines = self.SkelImage(imgSkel)
                            dicSkelLines["iInstIdx"] = iInstIdx
                            dicSkelLines["iSubInstIdx"] = iSubInstIdx
                            lSkelLines.append(dicSkelLines)
                        # endif
                    # endfor sub-instance
                # endif
            # endfor instance

            if bDoDrawPreview:
                for dicSkelLine in lSkelLines:
                    lSegRows = dicSkelLine.get("lSegRows")
                    lSegCols = dicSkelLine.get("lSegCols")

                    for iIdx in range(len(lSegRows)):
                        aRows = lSegRows[iIdx]
                        aCols = lSegCols[iIdx]

                        aPnts = np.c_[aCols, aRows].reshape((-1, 1, 2)).astype(np.int32)
                        imgTrgPreview = cv2.polylines(imgTrgPreview, aPnts, True, (0, 0, 255))
                    # endfor
                # endfor
            # endif do draw preview

            dicFlatType["lSkelLines"] = lSkelLines
        # endfor FlatType
        print("Skeletonize: finished", flush=True)
        print("", flush=True)

        return {"bOK": True, "bProcessed": True, "sMsg": ""}

    # enddef

    ###############################################################################
    def SkelImage(self, _imgSrc):
        try:
            from skimage.morphology import (
                skeletonize,
                medial_axis,
                closing,
                dilation,
                local_maxima,
            )
        except Exception:
            raise Exception("Module 'scikit-image' is not installed for calculating skeletons")
        # endif

        shpImg = _imgSrc.shape[0:2]
        imgSrcB = np.zeros(shpImg, dtype=np.uint8)
        imgSrcB[_imgSrc > 0] = 1

        imgSkel = np.zeros(shpImg, np.uint8)
        imgMax = np.zeros(shpImg, np.uint8)
        imgBox = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

        maskSkel, imgDist = medial_axis(imgSrcB, return_distance=True)

        maskEl = [[False, False, False], [True, True, True], [False, False, False]]
        maskMax1 = local_maxima(imgDist, maskEl)

        maskEl = [[False, True, False], [False, True, False], [False, True, False]]
        maskMax2 = local_maxima(imgDist, maskEl)

        maskEl = [[False, False, True], [False, True, False], [True, False, False]]
        maskMax3 = local_maxima(imgDist, maskEl)

        maskEl = [[True, False, False], [False, True, False], [False, False, True]]
        maskMax4 = local_maxima(imgDist, maskEl)

        imgCnt = np.zeros(shpImg, np.uint8)
        imgCnt[maskMax1] += 1
        imgCnt[maskMax2] += 1
        imgCnt[maskMax3] += 1
        imgCnt[maskMax4] += 1

        imgSkel.fill(0)
        imgSkel[imgCnt > 3] = 1

        imgMax.fill(0)
        imgMax[imgCnt > 2] = 1

        for iIdx in range(5):
            imgDil = dilation(imgSkel, footprint=imgBox)
            imgSkel = np.bitwise_and(imgDil, imgMax)
            imgSkel = closing(imgSkel, footprint=imgBox)
        # endfor

        # imgSkel = dilation(imgSkel, footprint=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # imgSkel = dilation(imgSkel, footprint=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        maskSkel = skeletonize(imgSkel)
        imgSkel = np.zeros(shpImg, np.uint8)
        imgSkel[maskSkel] = 1

        # order coordinates of skeleton pixels from start to end points
        aRows, aCols = np.nonzero(imgSkel)

        aSortIdx = np.flip(np.argsort(aRows))
        aRows = aRows[aSortIdx]
        aCols = aCols[aSortIdx]

        aSqRow = np.tile(aRows, (len(aRows), 1))
        aRowDist = np.abs(aSqRow - aSqRow.transpose())

        aSqCol = np.tile(aCols, (len(aCols), 1))
        aColDist = np.abs(aSqCol - aSqCol.transpose())

        aN = np.argwhere(np.logical_and(aRowDist <= 1, aColDist <= 1))
        aUnique, aCounts = np.unique(aN[:, 0], return_counts=True)

        aEndIdx = np.argwhere(aCounts == 2)
        # aN_forward = aN[aN[:, 0] < aN[:, 1]]
        # aN_fs = aN_forward[:, 0]
        aN0 = aN[:, 0]
        aN1 = aN[:, 1]

        lSegRows = []
        lSegCols = []
        lUsedEnds = []
        for lEndIdx in aEndIdx:
            iNextIdx = lEndIdx[0]
            if iNextIdx in lUsedEnds:
                continue
            # endif

            iCurIdx = iNextIdx
            iPrevIdx = iNextIdx
            lSeg = []
            while True:
                lSeg.append(iNextIdx)
                aIdx = aN1[aN0 == iNextIdx]
                iPrevIdx = iCurIdx
                iCurIdx = iNextIdx
                iNextIdx = next((x for x in aIdx if x != iPrevIdx and x != iCurIdx), None)
                if iNextIdx is None:
                    break
                # endif
            # endwhile
            lUsedEnds.append(iCurIdx)

            aSegRows = aRows[lSeg]
            aSegCols = aCols[lSeg]
            lSegRows.append(aSegRows.tolist())
            lSegCols.append(aSegCols.tolist())
        # endfor

        return {"lSegRows": lSegRows, "lSegCols": lSegCols}

    # enddef


# endclass
