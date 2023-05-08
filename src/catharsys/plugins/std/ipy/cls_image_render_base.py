#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \cls_cathipy_action.py
# Created Date: Tuesday, August 10th 2021, 3:41:16 pm
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
from typing import Optional, ForwardRef

from pathlib import Path
from IPython.display import Image, Video, JSON, display
from IPython.display import Markdown, HTML

from anybase.ipy import CIPyRenderBase
from catharsys.plugins.std.util import imgproc

TIPyImageRenderBase = ForwardRef("CIPyImageRenderBase")


def RenderImageExr(sFpImage: str, iWidth: int, iHeight: int):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    imgTrg = imgproc.LoadImageExr(sFpImage=sFpImage, bAsUint=True)
    _, imgPng = cv2.imencode(".png", imgTrg)

    return Image(data=imgPng, width=iWidth, height=iHeight)


# enddef


class CIPyImageRenderBase(CIPyRenderBase):

    ##########################################################################
    def __init__(self, _xIPyRenderBase: Optional[TIPyImageRenderBase] = None):
        super().__init__(_xIPyRenderBase)
        self.RegisterImageRenderer(sFileSuffix=".exr", funcImageRenderer=RenderImageExr)

    # enddef


# endclass
