#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \exec_python.py
# Created Date: Friday, May 6th 2022, 8:22:16 am
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

import sys
from xml.etree.ElementPath import xpath_tokenizer

from anybase.cls_python import CPythonConfig

if sys.version_info < (3, 10):
    import importlib_resources as res
else:
    from importlib import resources as res
# endif
import shutil
import platform
from pathlib import Path
from typing import Optional

import catharsys.plugins.std
import catharsys.util.version as cathversion
import catharsys.util.lsf as cathlsf
from catharsys.config.cls_project import CProjectConfig

from anybase import path as anypath
from anybase import assertion, convert, debug, config
from anybase.cls_any_error import CAnyError, CAnyError_Message, CAnyError_TaskMessage
from anybase.cls_process_handler import CProcessHandler
from anybase.cls_python import CPythonConfig

from catharsys.config.cls_exec_lsf import CConfigExecLsf
from .config.cls_exec_python import CConfigExecPython

from catharsys.decs.decorator_ep import EntryPoint
from catharsys.util.cls_entrypoint_information import CEntrypointInformation

from catharsys.action.cmd.ws_launch import NsKeys as NsLaunchKeys
from catharsys.decs.decorator_log import logFunctionCall


#########################################################################################################
@EntryPoint(CEntrypointInformation.EEntryType.EXE_PLUGIN)
def StartJob(*, xPrjCfg: CProjectConfig, dicExec: dict, dicArgs: dict):
    try:
        try:
            pathJobConfig = dicArgs["pathJobConfig"]
            sJobName = dicArgs["sJobName"]
            sJobNameLong = dicArgs["sJobNameLong"]
            dicTrial = dicArgs["dicTrial"]
            dicDebug = dicArgs["dicDebug"]
        except KeyError as xEx:
            raise CAnyError_Message(sMsg="Blender job argument '{}' missing".format(str(xEx)))
        # endtry

        xProcHandler: CProcessHandler = dicArgs.get("xProcessHandler")

        if sJobNameLong is None:
            sJobNameLong = sJobName
        # endif

        xExec = CConfigExecPython(dicExec)
        xLsf = CConfigExecLsf(dicExec)
        xPythonCfg = CPythonConfig(xPythonPath=xExec.pathPython, sCondaEnv=xExec.sCondaEnv)

        # xPythonCfg.ExecPython(lArgs=["--version"], bDoPrint=True, bDoPrintOnError=True)
        # pass

        # Since the typical demo "blender_auto.json5" execution file has type '*' in the
        # top DTI, we should interpret it as "std" if it is not overwritten by the
        # "__platform__" block.
        if xExec.sType == "std" or xExec.sType == "*":
            _StartPythonScript(
                xPythonCfg=xPythonCfg,
                pathJobConfig=pathJobConfig,
                bPrintOutput=True,
                dicDebug=dicDebug,
                xProcHandler=xProcHandler,
            )

        elif xExec.sType == "lsf":
            _LsfStartPythonScript(
                xPythonCfg=xPythonCfg,
                pathJobConfig=pathJobConfig,
                sJobName=sJobName,
                sJobNameLong=sJobNameLong,
                xLsfCfg=xLsf,
                bPrintOutput=True,
                xProcHandler=xProcHandler,
            )
        else:
            sExecFile = "?"
            if isinstance(dicExec, dict):
                sExecFile = config.GetDictValue(dicExec, "__locals__/filepath", str, bOptional=True, bAllowKeyPath=True)
            # endif

            raise CAnyError_Message(
                sMsg=(
                    f"Unsupported Python execution type '{xExec.sType}'. "
                    f"Have a look at your execution configuration file at: {sExecFile}.\n"
                    "Maybe the default execution 'sDTI' tag is not overwritten by "
                    "a matching '__platform__' block for your current platform."
                )
            )
        # endif

    except Exception as xEx:
        raise CAnyError_TaskMessage(sTask="Start Python Job", sMsg="Unable to start job", xChildEx=xEx)
    # endtry


# enddef


################################################################################################
# Start python script with standard suprocess call
def _StartPythonScript(
    *,
    xPythonCfg: CPythonConfig,
    pathJobConfig: Path,
    bPrintOutput: bool = True,
    dicDebug: dict = None,
    xProcHandler: Optional[CProcessHandler] = None,
):
    xScript = res.files(catharsys.plugins.std).joinpath("scripts").joinpath("run-action.py")
    with res.as_file(xScript) as pathScript:
        lScriptArgs = [pathScript.as_posix(), "--", pathJobConfig.as_posix()]
        if assertion.IsEnabled():
            lScriptArgs.append("--debug")
        # endif

        funcPostStart = None

        if dicDebug is not None:
            lScriptArgs.append("---")

            # calling cathy with e.g:  --script-vars dbg-port=portNumder option2=val2
            # every argument after --script-vars wil be split up and creates an
            # arg-pair -option val afterwards for the following script
            dicDebugScriptArgs = dicDebug.get(NsLaunchKeys.script_args, dict())
            if logFunctionCall.IsEnabled():
                dicDebugScriptArgs["log-call"] = "True"
            # endif

            for sKey, sValue in dicDebugScriptArgs.items():
                lScriptArgs.append(f"--{sKey}")
                if sValue is not None:  # it may happens and makes sense, that only keys are provided
                    lScriptArgs.append(sValue)
                # endif
            # endfor

            iDebugPort = dicDebug.get(NsLaunchKeys.iDebugPort)
            if iDebugPort is not None:
                lScriptArgs.extend(["--debug-port", f"{iDebugPort}"])
            else:
                bBreak = convert.DictElementToBool(dicDebugScriptArgs, "break", bDefault=False)
                if bBreak is True:
                    iDebugPort = convert.DictElementToInt(dicDebugScriptArgs, "dbg-port", iDefault=5678, bDoRaise=False)
                # endif
            # endif

            fDebugTimeout = convert.DictElementToFloat(dicDebug, NsLaunchKeys.fDebugTimeout, fDefault=10.0)

            bDebugSkipAction = dicDebug.get(NsLaunchKeys.bSkipAction)
            if bDebugSkipAction is True:
                lScriptArgs.extend(["--debug-skip-action", f"{bDebugSkipAction}"])
            # endif

            bShowGui = dicDebug.get(NsLaunchKeys.bShowGui)
            if bShowGui is True:
                lScriptArgs.extend(["--show-gui", f"{bShowGui}"])
            # endif

            if iDebugPort is not None:
                funcPostStart = debug.CreateHandler_CheckDebugPortOpen(
                    _fTimeoutSeconds=fDebugTimeout, _sIp="127.0.0.1", _iPort=iDebugPort
                )
            # endif

        # endif

        if xProcHandler is not None:
            xProcHandler.AddHandlerPostStart(funcPostStart)
        else:
            xProcHandler = CProcessHandler(_funcPostStart=funcPostStart)
        # endif

        iExitCode = xPythonCfg.ExecPython(
            lArgs=lScriptArgs,
            bDoPrint=bPrintOutput,
            bDoPrintOnError=bPrintOutput,
            xProcHandler=xProcHandler,
        )
    # endwith

    return {"bOK": True if iExitCode == 0 else False, "iExitCode": iExitCode}


# enddef


################################################################################################
# Start python script as LSF job
def _LsfStartPythonScript(
    *,
    xPythonCfg: CPythonConfig,
    pathJobConfig: Path,
    sJobName: str,
    sJobNameLong: str,
    xLsfCfg: CConfigExecLsf,
    bPrintOutput: bool = True,
    xProcHandler: Optional[CProcessHandler] = None,
):
    # Only supported on Linux platforms
    if platform.system() != "Linux":
        raise CAnyError_Message(sMsg="Unsupported system '{}' for LSF job creation".format(platform.system()))
    # endif

    sCondaEnv = xPythonCfg.sCondaEnv
    sSetCondaEnv = " "
    if sCondaEnv is not None:
        sSetCondaEnv = f"conda activate {sCondaEnv}"
    # endif

    sPathPython = xPythonCfg.sPathPython
    sSetPythonPath = " "
    if sPathPython is not None:
        if len(sPathPython) > 0:
            sSetPythonPath = "export PATH={0}:$PATH".format(sPathPython)
        # endif
    # endif

    ##################################################################################
    # Copy execution script to permament place from resources
    pathCathScripts = anypath.MakeNormPath("~/.catharsys/{}/scripts".format(cathversion.MajorMinorAsString()))
    pathCathScripts.mkdir(exist_ok=True, parents=True)
    pathBlenderScript = pathCathScripts / "run-action.py"

    xScript = res.files(catharsys.plugins.std).joinpath("scripts").joinpath("run-action.py")
    with res.as_file(xScript) as pathScript:
        shutil.copy(pathScript.as_posix(), pathBlenderScript.as_posix())
    # endwith
    ##################################################################################

    lArgs = [pathJobConfig.as_posix()]
    if assertion.IsEnabled():
        lArgs.append("--debug")
    # endif

    sScriptPars = " ".join(lArgs)
    sScriptFile = pathScript.as_posix()

    sScript = f"""
        mkdir lsf
        mkdir lsf/$LSB_BATCH_JID

                # Enable python either by loading the corresponding module
        # or by setting a path to a python install.
        {sSetCondaEnv}
        {sSetPythonPath}

        echo
        echo "Starting Standard rendering jobs..."
        echo

        echo "Script = {sScriptFile}"
        echo "Pars = {sScriptPars}"

        python {sScriptFile} -- {sScriptPars}
    """

    # print("Submitting job '{0}'...".format(sJobNameLong))

    bOk, lStdOut = cathlsf.Execute(
        _sJobName=sJobName,
        _xCfgExecLsf=xLsfCfg,
        _sScript=sScript,
        _bDoPrint=True,
        _bDoPrintOnError=True,
        _xProcHandler=xProcHandler,
    )

    return {"bOK": bOk, "sOutput": "\n".join(lStdOut)}


# enddef
