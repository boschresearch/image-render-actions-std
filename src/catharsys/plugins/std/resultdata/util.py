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
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /util.py
# Created Date: Tuesday, June 28th 2022, 4:34:56 pm
# Created by: Christian Perwass (CR/AEC5)
# -----
# Copyright (c) 2022 Robert Bosch GmbH and its subsidiaries
#
# All rights reserved.
# -----
###

import re
from pathlib import Path

from anybase import assertion


def AddFramesToDictFromPath(
    *,
    pathFrames: Path,
    dicImageFrames: dict,
    sFileExt: str,
    iFrameFirst: int,
    iFrameLast: int,
    iFrameStep: int,
) -> bool:
    assertion.FuncArgTypes()

    bHasImages = False

    if not sFileExt.startswith("."):
        sFileExt = "." + sFileExt
    # endif

    reImageFile = re.compile(f"Frame_(\\d+)\\{sFileExt}")
    for pathImage in pathFrames.iterdir():
        if not pathImage.is_file():
            continue
        # endif

        xMatch = reImageFile.match(pathImage.name)
        if xMatch is None:
            continue
        # endif

        iFrame = int(xMatch.group(1))
        if (
            iFrame < iFrameFirst
            or (iFrameLast >= 0 and iFrame > iFrameLast)
            or (iFrame - iFrameFirst) % iFrameStep != 0
        ):
            continue
        # endif

        bHasImages = True
        dicImageFrames[iFrame] = {"iFrameIdx": iFrame, "sFpImage": pathImage.as_posix()}
    # endfor frame image files

    return bHasImages


# enddef
