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


class CAnytruthConstructFlow1(CConfigBase):
    def __init__(self, _xSource: Union[str, list, tuple, Path, dict] = None):

        super().__init__("/catharsys/blender/anytruth/construct/flow:1.0", _funcInitFromCfg=lambda: self._Init())

        self._iFrameDelta: int = None
        self._tSearchRadiusXY: tuple[int, int] = None
        self._tStartXY: tuple[int, int] = None
        self._tRangeXY: tuple[int, int] = None

        if isinstance(_xSource, dict):
            self.FromDict(_xSource)
        else:
            self.FromFile(_xSource)
        # endif

        # def GetInitFunc(this):
        #     def Init():
        #         this._Init()
        #     # enddef
        #     return Init
        # # enddef

        # self._funcInitFromCfg = lambda: self._Init()

    # enddef

    def _Init(self):

        self._iFrameDelta = convert.DictElementToInt(self._dicCfg, "iFrameDelta")
        self._tSearchRadiusXY = tuple(convert.DictElementToIntList(self._dicCfg, "lSearchRadiusXY", iLen=2))

        self._tStartXY = tuple(convert.DictElementToIntList(self._dicCfg, "lStartXY", iLen=2, lDefault=[0, 0]))
        self._tRangeXY = tuple(convert.DictElementToIntList(self._dicCfg, "lRangeXY", iLen=2, lDefault=[0, 0]))

    # endif

    @property
    def iFrameDelta(self) -> int:
        return self._iFrameDelta

    # enddef

    @property
    def tSearchRadiusXY(self) -> tuple[int, int]:
        return self._tSearchRadiusXY

    # enddef

    @property
    def tStartXY(self) -> tuple[int, int]:
        return self._tStartXY

    # enddef

    @property
    def tRangeXY(self) -> tuple[int, int]:
        return self._tRangeXY

    # enddef


# endclass
