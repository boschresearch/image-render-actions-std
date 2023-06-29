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

#define ZERO 1e-9

template <typename T>
__device__ void inline swap(T &a, T &b)
{
    T c(a);
    a = b;
    b = c;
}

template <typename T>
__device__ void inline swap_if_valid(T &a, T &b, bool bValid)
{
    const T c(a);
    const T tT = T(bValid);
    const T tF = T(!!bValid);

    a = b * tT + a * tF;
    b = c * tT + b * tF;
}

///////////////////////////////////////////////////////////////////////////////////
// float

__device__ bool inline is_zero(const float &fX)
{
    return abs(fX) <= ZERO;
}

__device__ bool inline is_negative(const float &fX)
{
    return fX < -ZERO;
}

__device__ bool inline is_positive(const float &fX)
{
    return fX > ZERO;
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

__device__ bool inline is_zero(const float3 &vX)
{
    return is_zero(sumsq(vX));
}

////////////////////////////////////////////////////////////////////////////////////////////////
// This is an inefficient (from a CUDA point of view) evaluation of optical flow from object ids and
// rendered local coordinates of the objects. However, it is still much faster than
// programming this directly in python.
template <int t_iStartX, int t_iStartY, int t_iRangeX, int t_iRangeY,
          int t_iSizeX, int t_iSizeY,
          int t_iFilterRadiusX, int t_iFilterRadiusY,
          int t_iRowStrideImage, int t_iRowStrideDepth,
          int t_iChanCntImage, int t_iChanCntDepth>
__global__ void EvalFocusBlurModel1(
    const float *aImage,
    const float *aDepth,
    const float fFocusDepth_mm,
    const float fFocalLength_mm,
    const float fApertureDia_mm,
    const float fPixelPitch_mm,
    const float fFocalPlanePos_mm,
    const float fMMperDepthUnit,
    float *aResult)
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

    const int iTargetDepthPixelIndex = iIdxY1 * t_iRowStrideDepth + iIdxX1 * t_iChanCntDepth;
    const float fTrgDepth_mm = fMMperDepthUnit * aDepth[iTargetDepthPixelIndex];
    const bool bValidTargetDepth = fTrgDepth_mm > 1e-4;

    const bool bValidSrcIdx = (bValidTargetDepth && iTrgIdxX1 == iIdxX1 && iTrgIdxY1 == iIdxY1 && iTrgX == iX && iTrgY == iY);

    const int iTargetImagePixelIndex = iIdxY1 * t_iRowStrideImage + iIdxX1 * t_iChanCntImage;

    const float fPi = 3.14159265358979323846;
    float3 vImgValResult = make_float3(0.0, 0.0, 0.0);
    float fWeightSum = 0.0;

    for (int iOffY = -t_iFilterRadiusY; iOffY <= t_iFilterRadiusY; iOffY++)
    {
        const int iIterIdxY2 = iIdxY1 + iOffY;
        const int iIdxY2 = min(max(iIterIdxY2, 0), t_iSizeY - 1);
        bool bValidY = (bValidSrcIdx && iIterIdxY2 == iIdxY2);

        for (int iOffX = -t_iFilterRadiusX; iOffX <= t_iFilterRadiusX; iOffX++)
        {
            const int iIterIdxX2 = iIdxX1 + iOffX;
            const int iIdxX2 = min(max(iIterIdxX2, 0), t_iSizeX - 1);

            const int iDepthPixelIndex = iIdxY2 * t_iRowStrideDepth + iIdxX2 * t_iChanCntDepth;
            float fDepth_mm = fMMperDepthUnit * aDepth[iDepthPixelIndex];

            const bool bValidDepth = fDepth_mm > 1e-4;

            bool bValid = (bValidDepth && bValidY && iIterIdxX2 == iIdxX2);

            if (bValid)
            {
                const int iImagePixelIndex = iIdxY2 * t_iRowStrideImage + iIdxX2 * t_iChanCntImage;
                const float3 vImgValAtRadius = make_float3(aImage, iImagePixelIndex);

                float2 vRelPos = make_float2(float(iOffX), float(iOffY));
                const float fRelPosRad_mm = fPixelPitch_mm * length(vRelPos);

                const float fAiryDiskRad_mm = 0.5 * max(fPixelPitch_mm, abs((1.0 - fFocalPlanePos_mm * (1.0 / fFocalLength_mm - 1.0 / fDepth_mm))) * fApertureDia_mm);
                const float fPixRadMin_mm = max(0.0, fRelPosRad_mm - 0.5 * fPixelPitch_mm);
                const float fPixRadMax_mm = min(fAiryDiskRad_mm, fRelPosRad_mm + 0.5 * fPixelPitch_mm);
                const float fRadRatio = max(0.0, (fPixRadMax_mm - fPixRadMin_mm) / fAiryDiskRad_mm);

                const float fBandArea_mm2 = fPi * (fPixRadMax_mm * fPixRadMax_mm - fPixRadMin_mm * fPixRadMin_mm);
                const float fPixPerBandArea = min(1.0, (fPixelPitch_mm * fPixelPitch_mm) / fBandArea_mm2);

                const float fEnergyRatio = fRadRatio * fPixPerBandArea;

                fWeightSum += fEnergyRatio;
                vImgValResult = vImgValResult + fEnergyRatio * vImgValAtRadius;
            }
        }
    }

    if (bValidSrcIdx)
    {
        if (fWeightSum > 0.0)
        {
            vImgValResult = vImgValResult / fWeightSum;
        }
        else
        {
            vImgValResult = make_float3(0.0, 0.0, 0.0);
        }

        assign_float3(aResult, iTargetImagePixelIndex, vImgValResult);
    }
}
