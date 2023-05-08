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

////////////////////////////////////////////////////////////////////////////////////////////////
// This is an inefficient (from a CUDA point of view) evaluation of optical flow from object ids and
// rendered local coordinates of the objects. However, it is still much faster than
// programming this directly in python.
template <int t_iStartX, int t_iStartY, int t_iSizeX, int t_iSizeY,
          int t_iSearchRadiusX, int t_iSearchRadiusY,
          int t_iPosChanCnt, int t_iPosRowStride,
          int t_iIdxRowStride>
__global__ void EvalFlow(const float *aPos1, const float *aPos2, const int *aObjIdx1, const int *aObjIdx2, int *piIdxMapXY)
{
    const int iX = blockDim.x * blockIdx.x + threadIdx.x;
    const int iY = blockIdx.y;
    const int iTrgIdxX1 = iX + t_iStartX;
    const int iIdxX1 = min(max(iTrgIdxX1, 0), t_iSizeX - 1);
    const bool bValidSrcIdx = (iTrgIdxX1 == iIdxX1);

    const int iIdxY1 = iY + t_iStartY;

    const int iPosIdx1 = t_iPosRowStride * iIdxY1 + t_iPosChanCnt * iIdxX1;
    const int iPosObjIdx1 = t_iSizeX * iIdxY1 + iIdxX1;
    const int iObjIdx1 = aObjIdx1[iPosObjIdx1];

    float pfMinValue[3] = {1e9, 1e9, 1e9};
    int piMinPosIdx2[3] = {-1, -1, -1};

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
            for (int i = 0; i < 3; ++i)
            {
                const float fX = aPos1[iPosIdx1 + i] - aPos2[iPosIdx2 + i];
                fValue += fX * fX;
            }

            int iTempPosIdx2 = iPosIdx2;

            for (int i = 0; i < 3; ++i)
            {
                if (bValid && fValue < pfMinValue[i])
                {
                    swap(pfMinValue[i], fValue);
                    swap(piMinPosIdx2[i], iTempPosIdx2);
                }

                // bool bValid = (bValid && fValue < pfMinValue[i]);
                // // float fValid = float(iValid > 0 && fValue < pfMinValue[i]);
                // swap_if_valid(pfMinValue[i], fValue, bValid);
                // swap_if_valid(piMinPosIdx2[i], iTempPosIdx2, bValid);
            }
        }
    }

    // const int iValid = int(piMinPosIdx2[0] != piMinPosIdx2[1] && piMinPosIdx2[0] != piMinPosIdx2[2] && piMinPosIdx2[1] != piMinPosIdx2[2] && piMinPosIdx2[0] >= 0 && piMinPosIdx2[1] >= 0 && piMinPosIdx2[2] >= 0);

    int piX[3], piY[3];

    for (int i = 0; i < 3; ++i)
    {
        piX[i] = (piMinPosIdx2[i] % t_iPosRowStride) / t_iPosChanCnt;
        piY[i] = piMinPosIdx2[i] / t_iPosRowStride;
    }

    int iValid = int(bValidSrcIdx);
    // Check whether all indices have been set at all
    iValid = int(iValid > 0 && piMinPosIdx2[0] >= 0 && piMinPosIdx2[1] >= 0 && piMinPosIdx2[2] >= 0);
    // Check whether the index positions of the nearest three positions form a triangle with 1 pixel side lengths
    // iValid = int(iValid > 0 && abs(piX[0] - piX[1]) <= 1 && abs(piX[0] - piX[2]) <= 1 && abs(piX[1] - piX[2]) <= 1);
    // iValid = int(iValid > 0 && abs(piY[0] - piY[1]) <= 1 && abs(piY[0] - piY[2]) <= 1 && abs(piY[1] - piY[2]) <= 1);
    // iValid = int(iValid > 0 && (piX[0] != piX[1] || piX[0] != piX[2]));
    // iValid = int(iValid > 0 && (piY[0] != piY[1] || piY[0] != piY[2]));

    const int iIdxPos = iY * t_iIdxRowStride + iX * 6;
    if (iValid > 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            piIdxMapXY[iIdxPos + 2 * i + 0] = ((piMinPosIdx2[i] % t_iPosRowStride) / t_iPosChanCnt);
            piIdxMapXY[iIdxPos + 2 * i + 1] = (piMinPosIdx2[i] / t_iPosRowStride);
            // piIdxMapXY[iIdxPos + 2 * i + 0] = iValid * ((piMinPosIdx2[i] % t_iPosRowStride) / t_iPosChanCnt) + (iValid - 1);
            // piIdxMapXY[iIdxPos + 2 * i + 1] = iValid * (piMinPosIdx2[i] / t_iPosRowStride) + (iValid - 1);
        }
    }
    else if (bValidSrcIdx)
    {
        for (int i = 0; i < 3; ++i)
        {
            piIdxMapXY[iIdxPos + 2 * i + 0] = -1;
            piIdxMapXY[iIdxPos + 2 * i + 1] = -1;
        }
    }
}
