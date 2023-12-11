/*
 * File: \Dev-EvalFlow-v2.cu
 * Created Date: Tuesday, March 21st 2023, 8:04:17 am
 * Author: Christian Perwass (CR/AEC5)
 * <LICENSE id="Apache-2.0">
 *
 *   Image-Render Standard Actions module
 *   Copyright 2022 Robert Bosch GmbH and its subsidiaries
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * </LICENSE>
 */

#define ZERO 1e-12

template <typename T>
__device__ void inline swap(T &a, T &b)
{
    T c(a);
    a = b;
    b = c;
}

///////////////////////////////////////////////////////////////////////////////////
// float2

__device__ float2 inline make_float2(const float *pData, const int iIdx)
{
    float2 vX = {pData[iIdx], pData[iIdx + 1]};
    return vX;
}

__device__ float2 inline operator-(const float2 &vA, const float2 &vB)
{
    float2 vX = {vA.x - vB.x, vA.y - vB.y};
    return vX;
}

__device__ float2 inline operator+(const float2 &vA, const float2 &vB)
{
    float2 vX = {vA.x + vB.x, vA.y + vB.y};
    return vX;
}

__device__ float2 inline operator*(const float &fS, const float2 &vA)
{
    float2 vX = {fS * vA.x, fS * vA.y};
    return vX;
}

__device__ float2 inline operator*(const float2 &vA, const float &fS)
{
    float2 vX = {fS * vA.x, fS * vA.y};
    return vX;
}

__device__ float inline dot(const float2 &vX, const float2 &vY)
{
    return vX.x * vY.x + vX.y * vY.y;
}

__device__ float inline sumsq(const float2 &vX)
{
    return dot(vX, vX);
}

__device__ float inline length(const float2 &vX)
{
    return sqrt(sumsq(vX));
}

///////////////////////////////////////////////////////////////////////////////////
// float3
__device__ float3 inline make_float3(const float *pData)
{
    float3 vX = {pData[0], pData[1], pData[2]};
    return vX;
}

__device__ float3 inline make_float3(const float *pData, const int iIdx)
{
    float3 vX = {pData[iIdx], pData[iIdx + 1], pData[iIdx + 2]};
    return vX;
}

__device__ void inline assign_float3(float *pData, const int iIdx, const float3 &vX)
{
    pData[iIdx + 0] = vX.x;
    pData[iIdx + 1] = vX.y;
    pData[iIdx + 2] = vX.z;
}

__device__ float3 inline abs(const float3 &vX)
{
    float3 vY = {abs(vX.x), abs(vX.y), abs(vX.z)};
    return vY;
}

__device__ float3 inline max(const float3 &vX, const float3 &vY)
{
    float3 vZ = {max(vX.x, vY.x), max(vX.y, vY.y), max(vX.z, vY.z)};
    return vZ;
}

__device__ float inline dot(const float3 &vX, const float3 &vY)
{
    return vX.x * vY.x + vX.y * vY.y + vX.z * vY.z;
}

__device__ float inline sumsq(const float3 &vX)
{
    return dot(vX, vX);
}

__device__ float inline length(const float3 &vX)
{
    return sqrt(sumsq(vX));
}

__device__ float3 inline operator-(const float3 &vA, const float3 &vB)
{
    float3 vX = {vA.x - vB.x, vA.y - vB.y, vA.z - vB.z};
    return vX;
}

__device__ float3 inline operator+(const float3 &vA, const float3 &vB)
{
    float3 vX = {vA.x + vB.x, vA.y + vB.y, vA.z + vB.z};
    return vX;
}

__device__ float3 inline operator*(const float &fS, const float3 &vA)
{
    float3 vX = {fS * vA.x, fS * vA.y, fS * vA.z};
    return vX;
}

__device__ float3 inline operator*(const float3 &vA, const float &fS)
{
    float3 vX = {fS * vA.x, fS * vA.y, fS * vA.z};
    return vX;
}

__device__ float3 inline operator/(const float3 &vA, const float &fS)
{
    float3 vX = {vA.x / fS, vA.y / fS, vA.z / fS};
    return vX;
}

__device__ float3 inline operator/(const float3 &vA, const float3 &vB)
{
    float3 vX = {vA.x / vB.x, vA.y / vB.y, vA.z / vB.z};
    return vX;
}

__device__ bool inline is_zero(const float &fX)
{
    return abs(fX) <= ZERO;
}

__device__ bool inline is_zero(const float3 &vX)
{
    return is_zero(sumsq(vX));
}

__device__ bool inline is_negative(const float &fX)
{
    return fX < -ZERO;
}

__device__ bool inline is_positive(const float &fX)
{
    return fX > ZERO;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// This is an inefficient (from a CUDA point of view) evaluation of optical flow from object ids and
// rendered local coordinates of the objects. However, it is still much faster than
// programming this directly in python.
template <int t_iStartX, int t_iStartY, int t_iRangeX, int t_iRangeY,
          int t_iSizeX, int t_iSizeY,
          int t_iSearchRadiusX, int t_iSearchRadiusY,
          int t_iPosChanCnt, int t_iPosRowStride,
          int t_iIdxChanCnt, int t_iIdxRowStride,
          int t_iSubPixChanCnt, int t_iSubPixRowStride>
__global__ void EvalFlow(const float *aPos1, const float *aPos2, const int *aObjIdx1, const int *aObjIdx2, int *piIdxMapXY, float *pfSubPixIdx)
{
    const float fNaN = nanf("");

    const int iTrgX = blockDim.x * blockIdx.x + threadIdx.x;
    const int iX = min(iTrgX, t_iRangeX - 1);

    const int iTrgY = blockIdx.y;
    const int iY = min(iTrgY, t_iRangeY - 1);

    const int iTrgIdxX1 = iX + t_iStartX;
    const int iIdxX1 = min(max(iTrgIdxX1, 0), t_iSizeX - 1);

    const int iTrgIdxY1 = iY + t_iStartY;
    const int iIdxY1 = min(max(iTrgIdxY1, 0), t_iSizeY - 1);

    const int iPosIdx1 = t_iPosRowStride * iIdxY1 + t_iPosChanCnt * iIdxX1;
    const int iPosObjIdx1 = t_iSizeX * iIdxY1 + iIdxX1;
    const int iObjIdx1 = aObjIdx1[iPosObjIdx1];

    const bool bValidSrcIdx = (iObjIdx1 >= 0 && iTrgIdxX1 == iIdxX1 && iTrgIdxY1 == iIdxY1 && iTrgX == iX && iTrgY == iY);

    const float3 vPos1 = make_float3(aPos1, iPosIdx1);
    float3 vMaxDiff = make_float3(1e-3, 1e-3, 1e-3);

    // find the maximal gradient for each position dimension separately
    for (int iOffY = -1; iOffY <= 1; iOffY++)
    {
        const int iIterIdxY2 = iIdxY1 + iOffY;
        const int iIdxY2 = min(max(iIterIdxY2, 0), t_iSizeY - 1);
        const int iPosIdxY2 = t_iPosRowStride * iIdxY2;
        const bool bValidY = (bValidSrcIdx && iIterIdxY2 == iIdxY2);

        for (int iOffX = -1; iOffX <= 1; iOffX++)
        {
            const int iIterIdxX2 = iIdxX1 + iOffX;
            const int iIdxX2 = min(max(iIterIdxX2, 0), t_iSizeX - 1);

            const int iPosIdx2 = iPosIdxY2 + t_iPosChanCnt * iIdxX2;
            const int iPosObjIdx2 = t_iSizeX * iIdxY2 + iIdxX2;

            const bool bValid = (bValidY && iIterIdxX2 == iIdxX2 && iObjIdx1 == aObjIdx1[iPosObjIdx2]);

            const float3 vPos2 = make_float3(aPos1, iPosIdx2);
            const float3 vDiff = abs(vPos1 - vPos2);

            if (bValid)
            {
                vMaxDiff = max(vDiff, vMaxDiff);
            }
        }
    }

    vMaxDiff = vMaxDiff / max(vMaxDiff.x, max(vMaxDiff.y, vMaxDiff.z));

    float fMinValue = 1e38;
    int iMinPosIdx2 = -1;
    // int iMinObjIdx2 = -1;

    for (int iOffY = -t_iSearchRadiusY; iOffY <= t_iSearchRadiusY; iOffY++)
    {
        const int iIterIdxY2 = iIdxY1 + iOffY;
        const int iIdxY2 = min(max(iIterIdxY2, 0), t_iSizeY - 1);
        const int iPosIdxY2 = t_iPosRowStride * iIdxY2;
        bool bValidY = (bValidSrcIdx && iIterIdxY2 == iIdxY2);

        for (int iOffX = -t_iSearchRadiusX; iOffX <= t_iSearchRadiusX; iOffX++)
        {
            const int iIterIdxX2 = iIdxX1 + iOffX;
            const int iIdxX2 = min(max(iIterIdxX2, 0), t_iSizeX - 1);

            const int iPosIdx2 = iPosIdxY2 + t_iPosChanCnt * iIdxX2;
            const int iPosObjIdx2 = t_iSizeX * iIdxY2 + iIdxX2;

            bool bValid = (bValidY && iIterIdxX2 == iIdxX2 && iObjIdx1 == aObjIdx2[iPosObjIdx2]);

            float fValue = 0.0;
            float3 vPos2 = make_float3(aPos2, iPosIdx2);
            fValue = sqrt(sumsq((vPos1 - vPos2) / vMaxDiff));

            if (bValid && fValue < fMinValue)
            {
                fMinValue = fValue;
                iMinPosIdx2 = iPosIdx2;
                // iMinObjIdx2 = aObjIdx2[iPosObjIdx2];
            }
        }
    }

    const int iMapX = (iMinPosIdx2 % t_iPosRowStride) / t_iPosChanCnt;
    const int iMapY = iMinPosIdx2 / t_iPosRowStride;
    const int iMapIdxPos = iY * t_iIdxRowStride + iX * t_iIdxChanCnt;

    // Check whether all indices have been set at all
    bool bValid = bValidSrcIdx && iMinPosIdx2 >= 0;

    if (bValid)
    {
        const int iSubPixIdxPos = iY * t_iSubPixRowStride + iX * t_iSubPixChanCnt;
        const float3 vPosCtr = make_float3(aPos2, iMinPosIdx2);
        const float3 vVecTrg = vPos1 - vPosCtr;
        // float fIdxH = 0.0, fIdxV = 0.0;
        float2 vSubPix = make_float2(0.0, 0.0);
        bool bSubPixValid = true;

        if (!is_zero(vVecTrg))
        {
            bool pbValid[4];
            float3 pvPos[4];
            const int piH[4] = {1, 0, -1, 0};
            const int piV[4] = {0, -1, 0, 1};

            bSubPixValid = false;

            for (int i = 0; i < 4; i++)
            {
                const int iTrgMapX = iMapX + piH[i];
                const int iMapX = max(0, min(iTrgMapX, t_iSizeX - 1));

                const int iTrgMapY = iMapY + piV[i];
                const int iMapY = max(0, min(iTrgMapY, t_iSizeY - 1));

                const int iPosObjIdx = t_iSizeX * iMapY + iMapX;
                const int iPosIdx = t_iPosRowStride * iMapY + t_iPosChanCnt * iMapX;

                pbValid[i] = (iObjIdx1 == aObjIdx2[iPosObjIdx] && iMapX == iTrgMapX && iMapY == iTrgMapY);
                pvPos[i] = make_float3(aPos2, iPosIdx);
            }

            for (int i = 0; i < 4; i++)
            {
                const int iNext = (i + 1) % 4;
                const float3 vP1 = pvPos[i];
                const float3 vP2 = pvPos[iNext];

                if (pbValid[i] && pbValid[iNext])
                {
                    // Evaluate Barycentric Coordinates
                    const float3 vB1 = vP1 - vPosCtr;
                    const float3 vB2 = vP2 - vPosCtr;
                    const float2 vC = make_float2(dot(vVecTrg, vB1), dot(vVecTrg, vB2));
                    const float fB11 = dot(vB1, vB1);
                    const float fB12 = dot(vB1, vB2);
                    const float fB22 = dot(vB2, vB2);
                    const float fDetB = fB11 * fB22 - fB12 * fB12;
                    if (!is_zero(fDetB))
                    {
                        const float fL1 = (fB22 * vC.x - fB12 * vC.y) / fDetB;
                        const float fL2 = (fB11 * vC.y - fB12 * vC.x) / fDetB;
                        // It seems we do not always find the best central pixel.
                        // Therefore, fL1+fL2 can be greater than 1. Results look good nonetheless.
                        if ((fL1 >= 0.0 && fL2 >= 0.0) && (4.0 - fL1 - fL2) >= -1e-4)
                        {
                            vSubPix = fL1 * make_float2(float(piH[i]), float(piV[i])) + fL2 * make_float2(float(piH[iNext]), float(piV[iNext]));
                            // vSubPix = make_float2(fL1, fL2);
                            bSubPixValid = true;
                            break;
                        }
                    }
                }
            }
        }

        piIdxMapXY[iMapIdxPos + 0] = iObjIdx1;
        piIdxMapXY[iMapIdxPos + 1] = iIdxX1;
        piIdxMapXY[iMapIdxPos + 2] = iIdxY1;

        if (bSubPixValid)
        {
            const float fMapX = float(iMapX) + vSubPix.x;
            const float fMapY = float(iMapY) + vSubPix.y;

            pfSubPixIdx[iSubPixIdxPos + 0] = fMapX;
            pfSubPixIdx[iSubPixIdxPos + 1] = fMapY;
            pfSubPixIdx[iSubPixIdxPos + 2] = fMapX - float(iIdxX1);
            pfSubPixIdx[iSubPixIdxPos + 3] = fMapY - float(iIdxY1);
            // pfSubPixIdx[iSubPixIdxPos + 2] = vSubPix.x;
            // pfSubPixIdx[iSubPixIdxPos + 3] = vSubPix.y;

            piIdxMapXY[iMapIdxPos + 3] = iMapX;
            piIdxMapXY[iMapIdxPos + 4] = iMapY;
        }
        else
        {
            pfSubPixIdx[iSubPixIdxPos + 0] = fNaN;
            pfSubPixIdx[iSubPixIdxPos + 1] = fNaN;
            pfSubPixIdx[iSubPixIdxPos + 2] = fNaN;
            pfSubPixIdx[iSubPixIdxPos + 3] = fNaN;

            piIdxMapXY[iMapIdxPos + 3] = -1;
            piIdxMapXY[iMapIdxPos + 4] = -1;
        }
        // assign_float3(pfSubPixIdx, iIdxPos + 2, vPosCtr);
        // assign_float3(pfSubPixIdx, iIdxPos + 5, vPosR);

    } // if (bValid)
    else if (bValidSrcIdx)
    {
        piIdxMapXY[iMapIdxPos + 0] = -1;
        piIdxMapXY[iMapIdxPos + 1] = -1;
        piIdxMapXY[iMapIdxPos + 2] = -1;
        piIdxMapXY[iMapIdxPos + 3] = -1;
        piIdxMapXY[iMapIdxPos + 4] = -1;
        piIdxMapXY[iMapIdxPos + 5] = -1;
    }
}
