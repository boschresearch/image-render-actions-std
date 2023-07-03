#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cls_anytruth_construct_flow.py
# Created Date: Thursday, March 23rd 2023, 2:43:50 pm
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

from anybase.cls_config_base import CConfigBase

from typing import Union
from pathlib import Path

from anybase import path
from anybase import config
from anybase import convert
from anybase.cls_any_error import CAnyError_Message


class CConfigConstructFocusBlurModel1(CConfigBase):
    def __init__(self, _xSource: Union[str, list, tuple, Path, dict] = None):

        super().__init__("/catharsys/blender/construct/focus-blur/model1:1.0", _funcInitFromCfg=lambda: self._Init())

        self._tFilterRadiusXY: tuple[int, int] = None
        self._tStartXY: tuple[int, int] = None
        self._tRangeXY: tuple[int, int] = None
        self._sPathDepth: str = None
        self._sPathImage: str = None
        self._sImageFileExt: str = None
        self._fFocusDepth_mm: float = None
        self._fFocalLength_mm: float = None
        self._fApertureDia_mm: float = None
        self._fPixelPitch_mm: float = None
        self._fFocalPlanePos_mm: float = None
        self._fMMperDepthUnit: float = None
        self._fBackgroundDepth_mm: float = None

        if isinstance(_xSource, dict):
            self.FromDict(_xSource)
        else:
            self.FromFile(_xSource)
        # endif

    # enddef

    def _Init(self):

        self._tFilterRadiusXY = tuple(convert.DictElementToIntList(self._dicCfg, "lFilterRadiusXY", iLen=2))

        self._tStartXY = tuple(convert.DictElementToIntList(self._dicCfg, "lStartXY", iLen=2, lDefault=[0, 0]))
        self._tRangeXY = tuple(convert.DictElementToIntList(self._dicCfg, "lRangeXY", iLen=2, lDefault=[0, 0]))

        self._sPathDepth = convert.DictElementToString(self._dicCfg, "sPathDepth")
        self._sPathImage = convert.DictElementToString(self._dicCfg, "sPathImage")
        self._sImageFileExt = convert.DictElementToString(self._dicCfg, "sImageFileExt")

        self._fFocusDepth_mm = convert.DictElementToFloat(self._dicCfg, "fFocusDepth_mm")
        self._fFocalLength_mm = convert.DictElementToFloat(self._dicCfg, "fFocalLength_mm")
        self._fApertureDia_mm = convert.DictElementToFloat(self._dicCfg, "fApertureDia_mm")
        self._fPixelPitch_mm = convert.DictElementToFloat(self._dicCfg, "fPixelPitch_mm")
        self._fMMperDepthUnit = convert.DictElementToFloat(self._dicCfg, "fMMperDepthUnit")
        self._fBackgroundDepth_mm = convert.DictElementToFloat(self._dicCfg, "fBackgroundDepth_mm", fDefault=1e6)

    # endif

    @property
    def tFilterRadiusXY(self) -> tuple[int, int]:
        return self._tFilterRadiusXY

    # enddef

    @property
    def tStartXY(self) -> tuple[int, int]:
        return self._tStartXY

    # enddef

    @property
    def tRangeXY(self) -> tuple[int, int]:
        return self._tRangeXY

    # enddef

    @property
    def pathDepth(self) -> Path:
        return Path(self._sPathDepth)

    # enddef

    @property
    def pathImage(self) -> Path:
        return Path(self._sPathImage)

    # enddef

    @property
    def sImageFileExt(self) -> str:
        if not self._sImageFileExt.startswith("."):
            return f".{self._sImageFileExt}"
        # endif
        return self._sImageFileExt

    # enddef

    @property
    def fFocusDepth_mm(self) -> float:
        return self._fFocusDepth_mm

    # enddef

    @property
    def fFocalLength_mm(self) -> float:
        return self._fFocalLength_mm

    # enddef

    @property
    def fApertureDia_mm(self) -> float:
        return self._fApertureDia_mm

    # enddef

    @property
    def fFocalPlanePos_mm(self) -> float:
        return 1.0 / (1.0 / self._fFocalLength_mm - 1.0 / self._fFocusDepth_mm)

    # enddef

    @property
    def fPixelPitch_mm(self) -> float:
        return self._fPixelPitch_mm

    # enddef

    @property
    def fMMperDepthUnit(self) -> float:
        return self._fMMperDepthUnit

    # enddef

    @property
    def fBackgroundDepth_mm(self) -> float:
        return self._fBackgroundDepth_mm

    # enddef


# endclass
