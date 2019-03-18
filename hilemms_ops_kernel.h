#ifndef HILEMMS_OPS_KERNEL
#define HILEMMS_OPS_KERNEL
#include "hilemms.h"

using namespace std;

#ifdef OPS_2D
void KerSetCoordinates(const Real* coordX, const Real* coordY, const int* idx,
                       Real* coordinates) {
    coordinates[OPS_ACC_MD3(0, 0, 0)] = coordX[idx[0]];
    coordinates[OPS_ACC_MD3(1, 0, 0)] = coordY[idx[1]];
}
#endif  // OPS_2D

#ifdef OPS_3D
void KerSetCoordinates3D(const Real* coordX, const Real* coordY,
                         const Real* coordZ, const int* idx,
                         Real* coordinates) {
    coordinates[OPS_ACC_MD4(0, 0, 0, 0)] = coordX[idx[0]];
    coordinates[OPS_ACC_MD4(1, 0, 0, 0)] = coordY[idx[1]];
    coordinates[OPS_ACC_MD4(2, 0, 0, 0)] = coordZ[idx[2]];
}
#endif  // OPS_3D

// Kernel to set initial value for a particlaur component.
// This kernel is multicomponent ready.
void KerSetInitialMacroVarsHilemms(const Real* coordinates, const int* idx,
                                   Real* macroVars, Real* macroVarsInitVal,
                                   const int* componentId) {
#ifdef OPS_2D

    int compoIndex{*componentId};
    for (int m = VARIABLECOMPPOS[2 * compoIndex], i = 0;
         m <= VARIABLECOMPPOS[2 * compoIndex + 1]; m++, i++) {
        macroVars[OPS_ACC_MD2(m, 0, 0)] =
            macroVarsInitVal[OPS_ACC_MD3(i, 0, 0)];
    }
#endif

#ifdef OPS_3D

    int compoIndex{*componentId};

    for (int m = VARIABLECOMPPOS[2 * compoIndex], i = 0;
         m <= VARIABLECOMPPOS[2 * compoIndex + 1]; m++, i++) {
        macroVars[OPS_ACC_MD2(m, 0, 0, 0)] =
            macroVarsInitVal[OPS_ACC_MD3(i, 0, 0, 0)];
    }
#endif
}

#ifdef OPS_2D
void KerSetEmbeddedBodyBoundary(int* surfaceBoundary,
                               const int* geometryProperty, int* nodeType) {
    VertexGeometryTypes gp =
        (VertexGeometryTypes)geometryProperty[OPS_ACC1(0, 0)];
    VertexTypes vt = (VertexTypes)nodeType[OPS_ACC2(0, 0)];
    if (gp != VG_Fluid && gp != VG_ImmersedSolid && Vertex_Fluid != vt) {
        nodeType[OPS_ACC2(0, 0)] = *surfaceBoundary;
    }
}

void KerSetEmbeddedCircle(Real* diameter, Real* centerPos,
                         const Real* coordinates, int* nodeType,
                         int* geometryProperty) {
    if ((coordinates[OPS_ACC_MD3(0, 0, 0)] - centerPos[0]) *
                (coordinates[OPS_ACC_MD3(0, 0, 0)] - centerPos[0]) +
            (coordinates[OPS_ACC_MD3(1, 0, 0)] - centerPos[1]) *
                (coordinates[OPS_ACC_MD3(1, 0, 0)] - centerPos[1]) <=
        (*diameter) * (*diameter) / 4) {
        nodeType[OPS_ACC4(0, 0)] = (int)Vertex_ImmersedSolid;
        geometryProperty[OPS_ACC5(0, 0)] = (int)VG_ImmersedSolid;
    }
}

void KerSetEmbeddedEllipse(Real* semiMajorAxes, Real* semiMinorAxis,
                           Real* centerPos, const Real* coordinates,
                           int* nodeType, int* geometryProperty) {
    if ((coordinates[OPS_ACC_MD3(0, 0, 0)] - centerPos[0]) / (*semiMajorAxes) *
                (coordinates[OPS_ACC_MD3(0, 0, 0)] - centerPos[0]) /
                (*semiMajorAxes) +
            (coordinates[OPS_ACC_MD3(1, 0, 0)] - centerPos[1]) /
                (*semiMinorAxis) *
                (coordinates[OPS_ACC_MD3(1, 0, 0)] - centerPos[1]) /
                (*semiMinorAxis) <=
        1.0) {
        nodeType[OPS_ACC4(0, 0)] = (int)Vertex_ImmersedSolid;
        geometryProperty[OPS_ACC5(0, 0)] = (int)VG_ImmersedSolid;
    }
}

void KerSweep(const int* geometryProperty, int* nodeType) {
    VertexGeometryTypes vg =
        (VertexGeometryTypes)geometryProperty[OPS_ACC0(0, 0)];
    if (VG_ImmersedSolid == vg) {
        VertexGeometryTypes neiborVertexType[8];
        neiborVertexType[0] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(1, 0)];
        neiborVertexType[1] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(-1, 0)];
        neiborVertexType[2] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(0, 1)];
        neiborVertexType[3] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(0, -1)];
        neiborVertexType[4] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(1, 1)];
        neiborVertexType[5] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(-1, -1)];
        neiborVertexType[6] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(-1, 1)];
        neiborVertexType[7] =
            (VertexGeometryTypes)geometryProperty[OPS_ACC0(1, -1)];
        int fluidNeiborNum = 0;
        for (int i = 0; i < 8; i++) {
            if (VG_ImmersedSolid != neiborVertexType[i]) {
                fluidNeiborNum++;
            }
        }

        int solidNeiborNumatCoord{0};
        for (int i = 0; i < 4; i++) {
            if (VG_ImmersedSolid == neiborVertexType[i]) {
                solidNeiborNumatCoord++;
            }
        }
        if (fluidNeiborNum > 0 && solidNeiborNumatCoord <= 1) {
            nodeType[OPS_ACC1(0, 0)] = Vertex_Fluid;
            ops_printf(
                "A solid point is wiped off due to there are %d fluid points "
                "surrounded and only %d solid points at x and y coordinates\n ",
                fluidNeiborNum, solidNeiborNumatCoord);
        }
    }
}

void KerSyncGeometryProperty(const int* nodeType, int* geometryProperty) {
    VertexGeometryTypes gp =
        (VertexGeometryTypes)geometryProperty[OPS_ACC1(0, 0)];
    VertexTypes vt = (VertexTypes)nodeType[OPS_ACC0(0, 0)];
    if (Vertex_Fluid == vt && gp != VG_Fluid) {
        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_Fluid;
    }
}

void KerSetEmbeddedBodyGeometry(const int* nodeType, int* geometryProperty) {
    VertexTypes vt = (VertexTypes)nodeType[OPS_ACC1(0, 0)];
    if (Vertex_ImmersedSolid == vt) {
        VertexTypes neiborVertexType[8];
        /*
                        6*****2*****4
                        *     *     *
                        *     *     *
                        1***********0
                        *     *     *
                        *     *     *
                        5*****3*****7
        */
        neiborVertexType[0] = (VertexTypes)nodeType[OPS_ACC0(1, 0)];
        neiborVertexType[1] = (VertexTypes)nodeType[OPS_ACC0(-1, 0)];
        neiborVertexType[2] = (VertexTypes)nodeType[OPS_ACC0(0, 1)];
        neiborVertexType[3] = (VertexTypes)nodeType[OPS_ACC0(0, -1)];
        neiborVertexType[4] = (VertexTypes)nodeType[OPS_ACC0(1, 1)];
        neiborVertexType[5] = (VertexTypes)nodeType[OPS_ACC0(-1, -1)];
        neiborVertexType[6] = (VertexTypes)nodeType[OPS_ACC0(-1, 1)];
        neiborVertexType[7] = (VertexTypes)nodeType[OPS_ACC0(1, -1)];
        int fluidNeiborNum{0};
        for (int i = 0; i < 8; i++) {
            if (Vertex_ImmersedSolid != neiborVertexType[i]) {
                fluidNeiborNum++;
            }
        }
        int solidNeiborNumatCoord{0};
        for (int i = 0; i < 4; i++) {
            if (Vertex_ImmersedSolid == neiborVertexType[i]) {
                solidNeiborNumatCoord++;
            }
        }
        if (fluidNeiborNum > 0) {
            // outer corner
            if (2 == solidNeiborNumatCoord) {
                if ((Vertex_ImmersedSolid == neiborVertexType[0] &&
                     Vertex_ImmersedSolid == neiborVertexType[1]) ||
                    (Vertex_ImmersedSolid == neiborVertexType[0] &&
                     Vertex_ImmersedSolid == neiborVertexType[1])) {
                    ops_printf("%s\n",
                               "There appears to be hanged solid points,i.e., "
                               "the solid body may be too thin");
                }
                if (Vertex_ImmersedSolid == neiborVertexType[2] &&
                    Vertex_ImmersedSolid == neiborVertexType[1]) {
                    if (Vertex_ImmersedSolid == neiborVertexType[6]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IPJM_O;
                    } else {
                        ops_printf("%s\n",
                                   "There appears to be hanged solid points");
                    }
                }

                if (Vertex_ImmersedSolid == neiborVertexType[3] &&
                    Vertex_ImmersedSolid == neiborVertexType[1]) {
                    if (Vertex_ImmersedSolid == neiborVertexType[5]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IPJP_O;
                    } else {
                        ops_printf("%s\n",
                                   "There appears to be hanged solid points");
                    }
                }

                if (Vertex_ImmersedSolid == neiborVertexType[3] &&
                    Vertex_ImmersedSolid == neiborVertexType[0]) {
                    if (Vertex_ImmersedSolid == neiborVertexType[7]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IMJP_O;
                    } else {
                        ops_printf("%s\n",
                                   "There appears to be hanged solid points");
                    }
                }
                if (Vertex_ImmersedSolid == neiborVertexType[2] &&
                    Vertex_ImmersedSolid == neiborVertexType[0]) {
                    if (Vertex_ImmersedSolid == neiborVertexType[4]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IMJM_O;
                    } else {
                        ops_printf("%s\n",
                                   "There appears to be hanged solid points");
                    }
                }
            }
            // Planlar corner
            if (3 == solidNeiborNumatCoord) {
                if (Vertex_ImmersedSolid != neiborVertexType[0]) {
                    geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IP;
                }
                if (Vertex_ImmersedSolid != neiborVertexType[1]) {
                    geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IM;
                }
                if (Vertex_ImmersedSolid != neiborVertexType[2]) {
                    geometryProperty[OPS_ACC1(0, 0)] = (int)VG_JP;
                }
                if (Vertex_ImmersedSolid != neiborVertexType[3]) {
                    geometryProperty[OPS_ACC1(0, 0)] = (int)VG_JM;
                }
            }
            // Inner corner
            if (4 == solidNeiborNumatCoord) {
                if (1 == fluidNeiborNum) {
                    if (Vertex_ImmersedSolid != neiborVertexType[4]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IPJP_I;
                    }
                    if (Vertex_ImmersedSolid != neiborVertexType[5]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IMJM_I;
                    }
                    if (Vertex_ImmersedSolid != neiborVertexType[6]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IMJP_I;
                    }
                    if (Vertex_ImmersedSolid != neiborVertexType[7]) {
                        geometryProperty[OPS_ACC1(0, 0)] = (int)VG_IPJM_I;
                    }
                } else {
                    ops_printf("%s\n",
                               "There appears to be hanged solid points");
                }
            }
        }
    }
}
#endif  // End of OPS_2D

#endif  // HILEMMS_OPS_KERNEL