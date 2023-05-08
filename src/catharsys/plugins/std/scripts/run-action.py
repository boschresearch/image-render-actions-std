#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: \actions\run-action.py
# Created Date: Friday, April 23rd 2021, 9:51:24 am
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

from catharsys.config.cls_config_list import CConfigList
from anybase.cls_any_error import CAnyError, CAnyError_Message, CAnyError_TaskMessage
from anybase import plugin, assertion, debug

import catharsys.decs.decorator_log as logging_dec

# no abbreviation this time. use  logging_dec.logFunctionCall
# -> because behaviour change inside application script that is
#    ignored when using that abbreviation by importing such early
# from catharsys.decs.decorator_log import logFunctionCall

# __bDebug__ = True
__bDebug__ = False

bValidCall = False
sMsg = ""
sFpConfig = None


#######################################################################################
def NextCmdParam(_lArgs: list, _sCmdPattern: str) -> str:
    """looks for cmd in f_argv.
    It return a tuple of (bool, the string after the cmd pattern or None)"""
    if _sCmdPattern in _lArgs:
        iIdx = _lArgs.index(_sCmdPattern) + 1
        if len(_lArgs) >= iIdx:
            return (True, _lArgs[iIdx])
    # endif cmd
    return (False, None)


# end-def
#######################################################################################

if __bDebug__:
    # Don't write byte code next to source code
    sys.dont_write_bytecode = True
# endif


#######################################################################################
#######################################################################################
# run the script
#######################################################################################
try:

    bDebugMode = False

    lArgv: list[str] = sys.argv

    # print("sys.argv: {0}".format(lArgv))

    # -------------------------------------------------------------------------------------
    try:
        lArgv = lArgv[lArgv.index("--") + 1 :]
        if len(lArgv) >= 1:
            print("found config json file {0}".format(lArgv))
            bValidCall = True
        else:
            print("did not find config json file in arguments look into 'script_arg' {0}".format(lArgv))
            if "script_arg" in locals().keys():
                script_arg = locals().get("script_arg")
                lArgv.append(script_arg)
                print("found config json file in 'script_arg' {0}".format(script_arg))
                bValidCall = True
            else:
                sMsg = "Expect name of config json file"
            # endif script_arg
        # endif

    except ValueError:
        sMsg = "No or invalid command line arguments given."
    # endtry

    # -------------------------------------------------------------------------------------
    try:
        xCfg = CConfigList(lArgv=lArgv)
        xCfg.LoadFile(lArgv[0])
    except Exception as xEx:
        raise CAnyError_TaskMessage(sTask="Run Blender action", sMsg="Error", xChildEx=xEx)
    # endtry

    # prepare logging decorator
    # calling cathy with --script-vars log-call=True ?
    bLogCall, sLogCall = NextCmdParam(lArgv, "--log-call")
    if bLogCall and sLogCall.lower() == "true":
        logging_dec.SwitchLoggingOn(
            pathLogFile=str(xCfg.xPrjCfg.pathMain / "cathy.call.md"),
            sApplication="blender",
        )
    # endif

    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    bDebugMode = "--debug" in lArgv
    if bDebugMode is True:
        print("Enabling DEBUG mode")
    # endif
    assertion.Enable(bDebugMode)

    iDebugPort: int = None
    bDoDebugBreak: bool = False

    bDoDebugBreak, sDebugPort = NextCmdParam(lArgv, "--debug-port")
    if bDoDebugBreak:
        iDebugPort = int(sDebugPort)
        bDoDebugBreak = True

    else:
        # prepare vs-code break
        # calling cathy with --script-vars break=True ?
        bDoDebugBreak, sBreak = NextCmdParam(lArgv, "--break")
        if bDoDebugBreak:
            bDoDebugBreak = sBreak.lower() == "true"
        # endif

        # 5678 is the default attach port in the VS Code debug configurations.
        # Unless a host and port are specified, host defaults to 127.0.0.1
        # calling cathy with --script-vars dbg-port=portNumder ?
        iDebugPort = 5678
        bDebugPort, sDebugPort = NextCmdParam(lArgv, "--dbg-port")
        if bDebugPort:
            iDebugPort = int(sDebugPort)
        # endif
    # endif

    # -------------------------------------------------------------------------------------
    bRunAction = True
    if bDoDebugBreak:
        logging_dec.logFunctionCall.PrintLog("action waits for attaching debugger")
        debug.WaitForClient(iDebugPort)

        bRunAction = "--skip-action" not in lArgv and "--debug-skip-action" not in lArgv

    # endif bBreakFound
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # print(lArgv)
    # print(bDebugMode)

    if not bValidCall:
        logging_dec.logFunctionCall.LogDateTime("action-invalid Call")
        raise CAnyError_Message(sMsg="Invalid call: {0}".format(sMsg))
    # endif

    # When breaking, you can ignore the action, if you just want to test everything up to this point,
    # or in the case of a Blender action, just have Blender open. However, for the latter
    # you can also use the command 'cathy blender debug'.
    if bRunAction:
        try:

            # Look for action module
            epAction = plugin.SelectEntryPointFromDti(
                sGroup="catharsys.action",
                sTrgDti=xCfg.sActionDti,
                sTypeDesc="catharsys action module",
            )

            modAction = epAction.load()
            modAction.Run(xCfg)

        except Exception as xEx:
            logging_dec.logFunctionCall.LogDateTime("Run Action-Exception")
            raise CAnyError_TaskMessage(sTask=f"Running action '{xCfg.sActionDti}'", sMsg="Error", xChildEx=xEx)
        # endtry
    # endif bRunAction

    logging_dec.logFunctionCall.LogDateTime("Action")


except Exception as xEx:
    logging_dec.logFunctionCall.LogDateTime("Action execution script-exception")
    if isinstance(xEx, CAnyError):
        logging_dec.logFunctionCall.PrintLog("Report Any Error:")
        logging_dec.logFunctionCall.PrintLog(xEx.ToString())
    # end if
    CAnyError.Print(xEx, bTraceback=bDebugMode)
# endtry -> end of script

#######################################################################################
