#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cathy\cls_cfg_exec.py
# Created Date: Monday, April 26th 2021, 10:05:56 am
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
from pathlib import Path
from anybase import config
from anybase.cls_any_error import CAnyError_TaskMessage
from catharsys.setup import conda


class CConfigExecPython:
    ################################################################################
    def __init__(self, _dicExec):
        self._dicData = copy.deepcopy(_dicExec)
        dicResult = config.AssertConfigType(_dicExec, "/catharsys/exec/python/*:2.0")
        self._lType = dicResult["lCfgType"]

        self._dicPython = self._dicData.get("mPython", {})

    # enddef

    ################################################################################
    # Properties

    @property
    def dicPython(self) -> dict:
        return self._dicPython

    # enddef

    @property
    def sType(self) -> str:
        return self._lType[3]

    # enddef

    @property
    def pathPython(self) -> Path:
        sPyPath = self._dicPython.get("sPath")
        if sPyPath is None:
            return None
        # endif
        return Path(sPyPath)

    # enddef

    @property
    def sCondaEnv(self) -> str:
        return self._dicPython.get("sCondaEnv", conda.GetActiveEnvName())

    # enddef


# endclass
