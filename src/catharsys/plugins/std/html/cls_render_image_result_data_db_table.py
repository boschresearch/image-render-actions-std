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

import os
import sys
import base64
import hashlib

from typing import Optional, Union, ForwardRef, Callable
from pathlib import Path

# need to enable OpenExr explicitly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from catharsys.plugins.std.util.imgproc import LoadImageExr, UpdateWidthHeight

from anybase.cls_any_error import CAnyError_Message
from anybase import config
from anybase import path as anypath
from anybase.html import CHtmlPage, CHtmlTable
from catharsys.plugins.std.resultdata import CImageResultDataDb

THtmlRenderIRDDT = ForwardRef("CHtmlRenderImageResultDataDbTable")


########################################################################################
class CHtmlRenderImageResultDataDbTable:
    @property
    def bUseTrialIdAsPath(self):
        return self._bUseTrialIdAsPath

    @bUseTrialIdAsPath.setter
    def bUseTrialIdAsPath(self, bValue: bool):
        self._bUseTrialIdAsPath = bValue

    @property
    def bUseHashAsThumbFilename(self):
        return self._bUseHashAsThumbFilename

    @bUseHashAsThumbFilename.setter
    def bUseHashAsThumbFilename(self, bValue: bool):
        self._bUseHashAsThumbFilename = bValue

    ####################################################################################
    def __init__(self, _xData: CImageResultDataDb):
        self._xData = _xData

        self._pathStart: Path = self._xData.xProject.xConfig.pathOutput
        self._sFolderThumb: str = "thumbnails"
        self._sFolderOutput: str = self._xData.xProject.xConfig.sOutputFolderName
        self._pathOutput: Path = self._pathStart / self._xData.xProject.xConfig.sLaunchFolderName
        self._bUseTrialIdAsPath: bool = False
        self._bUseHashAsThumbFilename: bool = True

    # enddef

    ##########################################################################
    def SetOutputPathType(self, sWhere, sCustomPath=None, bCreatePath=False):

        if sWhere == "START" or sWhere == "LAUNCH":
            self._pathStart = self._xData.xProject.xConfig.pathLaunch
            if isinstance(sCustomPath, str):
                pathCustom = Path(sCustomPath)
                if pathCustom.is_absolute():
                    raise RuntimeError(f"Custom path must be a relative path for type '{sWhere}'")
                # endif
                self._pathOutput = anypath.MakeNormPath(self._pathStart / self._sFolderOutput / pathCustom)
            else:
                self._pathOutput = self._pathStart / self._sFolderOutput
            # endif

        elif sWhere == "WORKSPACE":
            self._pathStart = self._xData.xProject.xConfig.pathMain
            if isinstance(sCustomPath, str):
                pathCustom = Path(sCustomPath)
                if pathCustom.is_absolute():
                    raise RuntimeError(f"Custom path must be a relative path for type '{sWhere}'")
                # endif
                self._pathOutput = anypath.MakeNormPath(self._pathStart / self._sFolderOutput / pathCustom)
            else:
                self._pathOutput = self._pathStart / self._sFolderOutput
            # endif

        elif sWhere == "CUSTOM":
            if not isinstance(sCustomPath, str):
                raise RuntimeError("Given custom path is not a string")
            # endif

            pathCustom = anypath.MakeNormPath(sCustomPath)
            if bCreatePath is False and not pathCustom.exists():
                raise RuntimeError("Target output path does not exist")
            elif not pathCustom.exists():
                pathCustom.mkdir(parents=True)
            # endif
            self._pathStart = pathCustom
            self._pathOutput = self._pathStart / self._sFolderOutput
        else:
            raise RuntimeError("Invalid output path type. Allowed values are: 'START', 'WORKSPACE', 'CUSTOM'")
        # endif

    # enddef

    ####################################################################################
    def Render(self, **kwargs):
        sWhere = "Draw() function arguments"

        iImgWidth = config.GetDictValue(kwargs, "iImgWidth", int, xDefault=128, bOptional=True, sWhere=sWhere)
        iImgHeight = config.GetDictValue(kwargs, "iImgHeight", int, xDefault=None, bOptional=True, sWhere=sWhere)

        funcFilter = config.GetDictValue(kwargs, "funcFilter", function, xDefault=None, bOptional=True, sWhere=sWhere)
        bFramesAlongRows = config.GetDictValue(kwargs, "bFramesAlongRows", bool, xDefault=True, sWhere=sWhere)
        bOverwriteThumbnails = config.GetDictValue(kwargs, "bOverwriteThumbnails", bool, xDefault=False, sWhere=sWhere)
        bCreateLinks = config.GetDictValue(kwargs, "bCreateLinks", bool, xDefault=True, sWhere=sWhere)

        iConfigsPerPage = config.GetDictValue(
            kwargs, "iConfigsPerPage", int, xDefault=None, bOptional=True, sWhere=sWhere
        )
        sName = config.GetDictValue(kwargs, "sName", str, xDefault="", sWhere=sWhere)

        self.RenderContactSheets(
            iImgWidth=iImgWidth,
            iImgHeight=iImgHeight,
            funcFilter=funcFilter,
            bFramesAlongRows=bFramesAlongRows,
            bOverwriteThumbnails=bOverwriteThumbnails,
            bCreateLinks=bCreateLinks,
            iConfigsPerPage=iConfigsPerPage,
            sName=sName,
        )

    # enddef

    ####################################################################################
    def RenderContactSheets(
        self,
        *,
        iImgWidth: int = 128,
        iImgHeight: Optional[int] = None,
        funcFilter: Optional[Callable] = None,
        bFramesAlongRows: bool = True,
        bOverwriteThumbnails: bool = False,
        bCreateLinks: bool = True,
        iConfigsPerPage: Optional[int] = None,
        sName: str = "",
    ) -> Path:

        dicTrialTables = self.CreateTrialTable(funcFilter=funcFilter)
        self.CreateThumbnails(
            dicTrialTables=dicTrialTables,
            iImgWidth=iImgWidth,
            iImgHeight=iImgHeight,
            bOverwrite=bOverwriteThumbnails,
        )

        xHtml = CHtmlPage()
        pathHtml = None

        # Create display table output
        for sTrialId in dicTrialTables:

            if self._bUseTrialIdAsPath:
                pathHtml = self._pathOutput / sTrialId
            else:
                pathHtml = self._pathOutput
            # endif

            dicHtmlTables = self.RenderTrialTable(
                dicTrialTables,
                iImgWidth=iImgWidth,
                iImgHeight=iImgHeight,
                bUseThumbnails=True,
                bCreateLinks=bCreateLinks,
                bFramesAlongRows=bFramesAlongRows,
                pathHtml=pathHtml,
                sSelectedTrialId=sTrialId,
            )

            dicTable: dict = dicTrialTables[sTrialId]
            xHtmlTable: CHtmlTable = dicHtmlTables[sTrialId]

            iCfgCnt = xHtmlTable.GetRowCnt() - 1
            iCfgPerPage = iConfigsPerPage if iConfigsPerPage is not None else iCfgCnt
            iPageCnt = int(iCfgCnt / iCfgPerPage) + (1 if iCfgCnt % iCfgPerPage > 0 else 0)
            iCfgStart = 0

            sBaseConfig = dicTable["sBaseConfig"]

            lPageFilenames = []
            for iPageIdx in range(iPageCnt):
                sPageFilename = ""
                if len(sName) > 0:
                    sPageFilename += sName + "_"
                # endif
                if self._bUseTrialIdAsPath is False:
                    sPageFilename += self._ToFilename(sTrialId) + "_"
                # endif
                sPageFilename += "page_{:02d}_of_{:02d}.html".format(iPageIdx + 1, iPageCnt)
                lPageFilenames.append(sPageFilename)
            # endfor

            for iPageIdx in range(iPageCnt):
                xHtml.Clear()

                if iPageCnt > 1:
                    lPageLinks = []
                    for iIdx in range(iPageCnt):
                        if iIdx == iPageIdx:
                            sLink = f"<bold>{iIdx+1}</bold>"
                        else:
                            sLink = '<a href="./{}">{}</a>'.format(lPageFilenames[iIdx], iIdx + 1)
                        # endif
                        lPageLinks.append(sLink)
                    # endfor
                    sPageLinks = " ".join(lPageLinks)
                    sStepLinks = ""
                    if iPageIdx > 0:
                        sStepLinks += '<a href="{}">previous</a> '.format(lPageFilenames[iPageIdx - 1])
                    # endif
                    if iPageIdx < iPageCnt - 1:
                        sStepLinks += '<a href="{}">next</a> '.format(lPageFilenames[iPageIdx + 1])
                    # endif
                    xHtml.Paragraph(f"Pages: {sPageLinks}<br>\n{sStepLinks}\n<hr>")

                # endif

                xHtml.Header(f"Trial: <code>{sTrialId}</code>", 1)
                if len(sBaseConfig) > 0:
                    xHtml.Line(f"<bold>Base Config</bold>: <code>{sBaseConfig}</code><br>")
                # endif

                if iPageCnt > 1:
                    xHtml.Line("<bold>Page {}/{}</bold><br>".format(iPageIdx + 1, iPageCnt))
                # endif

                xHtml.Paragraph(xHtmlTable.RenderTable(iRowStart=iCfgStart, iRowCount=iCfgPerPage))

                if iPageCnt > 1:
                    xHtml.Paragraph(f"<hr>\n{sStepLinks}<br>\nPages: {sPageLinks}")
                # endif

                sFpHtml = pathHtml / lPageFilenames[iPageIdx]

                with open(sFpHtml, "w") as xFile:
                    xFile.write(xHtml.Render())
                # endwith

                iCfgStart += iCfgPerPage
            # endfor
        # endfor

        return pathHtml

    # enddef

    ##########################################################################
    def CreateTrialTable(self, *, funcFilter=None) -> dict:

        dicState = {
            "TRIAL": None,
            "CONFIG": None,
            "RENDER_TYPE": None,
            "RENDER_OUT_TYPE": None,
            "FRAME": None,
        }

        dicTrialTables = {}
        for sTrialId in self._xData.dicImages:

            if funcFilter is not None:
                dicState["TRIAL"] = sTrialId
                if not funcFilter(sType="TRIAL", sValue=sTrialId, dicState=dicState):
                    continue
                # endif
            # endif

            dicTrialDb = self._xData.dicImages[sTrialId]

            dicTable = dicTrialTables[sTrialId] = {}
            dicHeader = dicTable["mHeader"] = {"Config": ["Config"]}
            lRows = dicTable["lRows"] = []
            lConfigs = []

            for sCfgId in dicTrialDb:
                if funcFilter is not None:
                    dicState["CONFIG"] = sCfgId
                    if not funcFilter(sType="CONFIG", sValue=sCfgId, dicState=dicState):
                        continue
                    # endif
                # endif
                dicCfg = dicTrialDb[sCfgId]

                dicRow = {}
                dicRow["Config"] = sCfgId
                lConfigs.append(sCfgId.split("/"))

                for sRndType in dicCfg:
                    if funcFilter is not None:
                        dicState["RENDER_TYPE"] = sRndType
                        if not funcFilter(sType="RENDER_TYPE", sValue=sRndType, dicState=dicState):
                            continue
                        # endif
                    # endif
                    dicRndType = dicCfg[sRndType]
                    sSubPathCfg = dicRndType["sSubPathCfg"]
                    dicRndOutTypes = dicRndType["mOutputType"]
                    for sRndOutType in dicRndOutTypes:
                        if funcFilter is not None:
                            dicState["RENDER_OUT_TYPE"] = sRndOutType
                            if not funcFilter(
                                sType="RENDER_OUT_TYPE",
                                sValue=sRndOutType,
                                dicState=dicState,
                            ):
                                continue
                            # endif
                        # endif
                        dicRndOutType = dicRndOutTypes[sRndOutType]

                        dicFrames = dicRndOutType["mFrames"]
                        dicImages = {}
                        for sFrame in dicFrames:
                            if funcFilter is not None:
                                dicState["FRAME"] = sFrame
                                if not funcFilter(sType="FRAME", sValue=sFrame, dicState=dicState):
                                    continue
                                # endif
                            # endif

                            dicFra = dicFrames[sFrame]
                            if isinstance(dicFra.get("sFpImage"), str):
                                dicImages[sFrame] = dicFra["sFpImage"]
                            elif isinstance(dicFra.get("lFpSubImages"), list):
                                dicImages[sFrame] = dicFra["lFpSubImages"]
                            else:
                                dicImages[sFrame] = None
                            # endif

                            sHeaderId = "{}/{}/{}".format(sRndType, sSubPathCfg, sRndOutType)
                            dicHeader[sHeaderId] = [sRndType, sSubPathCfg, sRndOutType]

                            dicRow[sHeaderId] = dicImages
                        # endfor frame
                        dicState["FRAME"] = None

                    # endfor render output type
                    dicState["RENDER_OUT_TYPE"] = None

                # endfor render type
                dicState["RENDER_TYPE"] = None

                lRows.append(dicRow)
            # endfor configs
            dicState["CONFIG"] = None

            if len(lConfigs) > 0:
                lCommon = []
                lParts = lConfigs[0]
                for iPartIdx, sPart in enumerate(lParts):
                    if all((x[iPartIdx] == sPart for x in lConfigs)):
                        lCommon.append(sPart)
                    else:
                        break
                    # endif
                # endfor parts

                iComCnt = len(lCommon)
                if iComCnt > 0:
                    for iRowIdx, dicRow in enumerate(lRows):
                        dicRow["Config"] = "/".join(lConfigs[iRowIdx][iComCnt:])
                    # endfor
                # endif
            # endif

            dicTable["sBaseConfig"] = "/".join(lCommon)
        # endfor trials

        return dicTrialTables

    # enddef

    ##########################################################################
    def RenderTrialTable(
        self,
        dicTrialTables,
        iImgWidth=None,
        iImgHeight=None,
        bUseThumbnails=False,
        bCreateLinks=True,
        bFramesAlongRows=False,
        pathHtml=None,
        sSelectedTrialId=None,
    ) -> dict:

        dicTableStyle = {
            "sStyleColHeader": "text-align:left",
            "sStyleRowHeader": "text-align:left",
            "bHasColHeader": True,
            "bHasRowHeader": True,
        }

        if pathHtml is None:
            pathHtml = self._pathStart
        # endif

        sStyleImage = ""
        if iImgWidth is not None:
            sStyleImage += "width:{0}px;min-width:{0}px;".format(iImgWidth)
        # endif
        if iImgHeight is not None:
            sStyleImage += "height:{0}px;min-height:{0}px;".format(iImgHeight)
        # endif

        # Create display table output
        dicHtmlTables = {}
        for sTrialId in dicTrialTables:
            if sSelectedTrialId is not None and sTrialId != sSelectedTrialId:
                continue
            # endif

            dicTable = dicTrialTables[sTrialId]
            sBaseConfig = dicTable["sBaseConfig"]

            dicHeader = dicTable["mHeader"]
            lCfgRows = dicTable["lRows"]

            if len(dicHeader) == 1:
                raise CAnyError_Message(sMsg=f"No images in trial '{sTrialId}'")
            # endif

            lImgHeader = []
            lImgHeadId = []
            iMaxImgCnt = 0
            sMaxImgId = "Config"
            for sHeaderId in dicHeader:
                if sHeaderId == "Config":
                    continue
                # endif
                lHeadText = [x for x in dicHeader[sHeaderId] if x != "" and x != "."]
                sHeadText = "<br>".join(lHeadText)
                lImgHeader.append(sHeadText)
                lImgHeadId.append(sHeaderId)

                iImgCnt = len(lCfgRows[0].get(sHeaderId, 0))
                if iImgCnt > iMaxImgCnt or iMaxImgCnt == 0:
                    iMaxImgCnt = iImgCnt
                    sMaxImgId = sHeaderId
                # endif
            # endfor header
            iImgTypeCnt = len(lImgHeader)
            dicImages = lCfgRows[0].get(sMaxImgId)
            lImageId = list(dicImages.keys())
            lImageId.sort()

            dicImgTableStyle = {
                "sStyleColHeader": "text-align:left",
                "sStyleRowHeader": "text-align:left",
                "bHasColHeader": True,
                "bHasRowHeader": True,
            }

            if iMaxImgCnt == 1:
                bFramesAlongRows = False
            # endif

            bUseSingleTable = bFramesAlongRows is False and iMaxImgCnt == 1

            if bUseSingleTable:
                xTableMain = CHtmlTable(
                    iRows=len(lCfgRows) + 1,
                    iCols=len(dicHeader),
                    dicStyle=dicTableStyle,
                )

                for iIdx, sHead in enumerate(lImgHeader):
                    xTableMain.AddHtml(0, iIdx + 1, sHead)
                # endfor
            else:
                xTableMain = CHtmlTable(iRows=len(lCfgRows) + 1, iCols=2, dicStyle=dicTableStyle)
            # endif

            xTableMain.AddHtml(0, 0, dicHeader["Config"][0])

            for iCfgRowIdx, dicCfgRow in enumerate(lCfgRows):
                # Get config header name
                sHeaderId = dicHeader["Config"][0]
                xTableMain.AddHtml(iCfgRowIdx + 1, 0, dicCfgRow.get(sHeaderId).replace("/", "<br>/"))

                if not bUseSingleTable:
                    xTableImgs = CHtmlTable(
                        iRows=iMaxImgCnt + 1,
                        iCols=iImgTypeCnt + 1,
                        dicStyle=dicImgTableStyle,
                    )
                    # first row are the image types
                    for iIdx, sHead in enumerate(lImgHeader):
                        xTableImgs.AddHtml(0, iIdx + 1, sHead)
                    # endfor

                    # first column are the frame indices
                    for iIdx, sImgId in enumerate(lImageId):
                        xTableImgs.AddHtml(iIdx + 1, 0, sImgId)
                    # endfor
                # endif

                for iImgColIdx, sHeaderId in enumerate(lImgHeadId):
                    dicImages = dicCfgRow.get(sHeaderId)
                    if dicImages is None:
                        continue
                    # endif

                    for iImgRowIdx, sImgId in enumerate(lImageId):

                        sHtml = ""

                        sFpImage = dicImages[sImgId]
                        if isinstance(sFpImage, str):
                            pathImage = Path(sFpImage)
                            sFpImage = pathImage.as_posix()
                            pathRelImg = Path(os.path.relpath(pathImage.as_posix(), pathHtml.as_posix()))
                            # pathRelImg = pathImage.relative_to(pathHtml)
                            # pathRelImg = pathImage.relative_to(self.pathStart)
                            sPathRelImg = "./{}".format(pathRelImg.as_posix())

                            if bCreateLinks is True:
                                sHtml += '<a href="{}" target="_blank">'.format(sPathRelImg)
                            # endif

                            if sFpImage.endswith(".exr") and not bUseThumbnails:
                                sHtml += "Image"

                            elif bUseThumbnails:
                                pathThumbImage = self._GetThumbnailFilepath(
                                    sTrialId,
                                    sBaseConfig,
                                    dicCfgRow["Config"],
                                    sHeaderId,
                                    sImgId,
                                    iImgWidth=iImgWidth,
                                    iImgHeight=iImgHeight,
                                )
                                sPathRelThumbImg = "./" + pathThumbImage.relative_to(pathHtml).as_posix()

                                sHtml += '<img src="{}" alt="{}" style="{}"/>'.format(
                                    sPathRelThumbImg, sFpImage, sStyleImage
                                )

                            else:
                                sHtml += '<img src="{}" alt="{}" style="{}"/>'.format(
                                    sPathRelImg, sFpImage, sStyleImage
                                )
                            # endif

                            if bCreateLinks:
                                sHtml += "</a>"
                            # endif
                        elif isinstance(sFpImage, list):
                            sHtml = "Image List"
                        else:
                            sHtml = "n/a"
                        # endif

                        if bUseSingleTable:
                            xTableMain.AddHtml(iCfgRowIdx + 1, iImgColIdx + 1, sHtml)
                        else:
                            xTableImgs.AddHtml(iImgRowIdx + 1, iImgColIdx + 1, sHtml)
                        # endif
                    # endfor image frames
                # endfor header id

                if not bUseSingleTable:
                    xTableImgs.SetTranspose(bFramesAlongRows)
                    bShowImgColHeader = iMaxImgCnt > 1 or (iMaxImgCnt == 1 and iCfgRowIdx == 0)

                    sTableHtml = xTableImgs.RenderTable(
                        bShowColHeader=bShowImgColHeader,
                        bShowRowHeader=(iMaxImgCnt > 1),
                        iIndent=xTableMain.GetIndentLen(),
                    )
                    xTableMain.AddHtml(iCfgRowIdx + 1, 1, sTableHtml)
                # endif

            # endfor rows
            dicHtmlTables[sTrialId] = xTableMain
        # endfor trials

        return dicHtmlTables

    # enddef

    ##########################################################################
    def CreateThumbnails(
        self,
        dicTrialTables,
        iImgWidth=128,
        iImgHeight=128,
        bNormalize=True,
        xFilter=None,
        bOverwrite=True,
    ):

        if (iImgWidth <= 0 or iImgWidth is None) and (iImgHeight <= 0 or iImgHeight is None):
            raise RuntimeError("Given thumbnail width and height are both invalid")
        # endif

        lThumbs = []

        iTrialCnt = len(dicTrialTables)
        for iTrialIdx, sTrialId in enumerate(dicTrialTables):
            pathThumb = self._GetThumbnailPath(sTrialId)
            pathThumb.mkdir(parents=True, exist_ok=True)

            dicTable = dicTrialTables[sTrialId]
            sBaseConfig = dicTable["sBaseConfig"]

            dicHeader = dicTable["mHeader"]
            lRows = dicTable["lRows"]

            iRowCnt = len(lRows)
            for iRowIdx, dicRow in enumerate(lRows):
                sys.stdout.write(
                    "Scanning for thumbnails: trial {}/{}, row {}/{}\r".format(
                        iTrialIdx + 1, iTrialCnt, iRowIdx + 1, iRowCnt
                    )
                )
                sys.stdout.flush()

                for sHeaderId in dicHeader:
                    if sHeaderId != "Config":
                        dicImages = dicRow.get(sHeaderId)
                        if dicImages is not None:
                            for sImgId in dicImages:
                                pathThumbImage = self._GetThumbnailFilepath(
                                    sTrialId,
                                    sBaseConfig,
                                    dicRow["Config"],
                                    sHeaderId,
                                    sImgId,
                                    iImgWidth=iImgWidth,
                                    iImgHeight=iImgHeight,
                                )
                                if bOverwrite is False and pathThumbImage.exists():
                                    continue
                                # endif

                                sFpImage = dicImages[sImgId]
                                if isinstance(sFpImage, str):
                                    lThumbs.append((sFpImage, pathThumbImage.as_posix()))
                                # endif

                            # endfor images
                        # endif has images
                    # endif not config header
                # endfor header ids
            # endfor rows
        # endfor trials

        sys.stdout.write("                                                                                   \r")
        sys.stdout.flush()

        iCnt = len(lThumbs)
        for iIdx, tThumb in enumerate(lThumbs):

            sys.stdout.write("Creating thumbnails {:5.2f}%\r".format(100.0 * (iIdx / iCnt)))
            sys.stdout.flush()

            sFpImage, sFpThumb = tThumb

            if ".exr" in sFpImage:
                aImage = LoadImageExr(sFpImage=sFpImage, bAsUint=True, bNormalize=bNormalize)
            else:
                aImage = cv2.imread(sFpImage)
            # endif

            if xFilter is not None:
                aImage = xFilter(aImage)
            # endif

            iW, iH = UpdateWidthHeight(iImgWidth, iImgHeight, aImage)
            # print(f"{iH}, {iW}")
            aImage = cv2.resize(aImage, (iW, iH))

            cv2.imwrite(sFpThumb, aImage)
        # endfor

        sys.stdout.write("                                                                                   \r")
        sys.stdout.flush()

    # enddef

    ##########################################################################
    def _ToFilename(self, _xName):

        sName = str(_xName)

        lReplaceList = [
            ["/./", "_"],
            ["\\.\\", "_"],
            ["\\", "_"],
            ["/", "_"],
            [" ", "_"],
        ]
        for lR in lReplaceList:
            sName = sName.replace(lR[0], lR[1])
        # endfor

        return sName

    # enddef

    ##########################################################################
    def _GetThumbnailFilename(self, _lIds, sExtension="png"):

        lNames = [self._ToFilename(x) for x in _lIds]
        sName = "-".join(lNames)
        if self._bUseHashAsThumbFilename:
            hashMD5 = hashlib.md5(sName.encode("utf8"))
            sName = base64.b32encode(hashMD5.digest()).decode()
            # iHash = hash(sName)
            # sName = "{:X}".format(iHash)
        # endif

        sFilename = "{}.{}".format(sName, sExtension)

        return sFilename

    # enddef

    ##########################################################################
    def _GetThumbnailPath(self, sTrialId):

        if self._bUseTrialIdAsPath is True:
            pathThumb = self._pathOutput / sTrialId / self._sFolderThumb
        else:
            pathThumb = self._pathOutput / self._sFolderThumb
        # endif
        return pathThumb

    # enddef

    ##########################################################################
    def _GetThumbnailFilepath(
        self,
        sTrialId,
        sBaseCfgId,
        sCfgId,
        sHeaderId,
        sImgId,
        iImgWidth=None,
        iImgHeight=None,
        sExtension="png",
    ):

        lIds = [sBaseCfgId, sCfgId, sHeaderId, sImgId]
        if self._bUseTrialIdAsPath is False:
            lIds.insert(0, sTrialId)
        # endif
        if iImgWidth is not None:
            lIds.append(f"w{iImgWidth}")
        # endif
        if iImgHeight is not None:
            lIds.append(f"h{iImgHeight}")
        # endif

        sFilename = self._GetThumbnailFilename(lIds, sExtension=sExtension)
        pathThumb = self._GetThumbnailPath(sTrialId)
        pathThumb = pathThumb / sFilename
        return pathThumb

    # enddef


# endclass
