#include <vector>
#include "flowfield.h"
#include "flowfield_host_device.h"
#include "boundary.h"
#include "model.h"
#include "scheme.h"
#include "ops_seq_v2.h"
#include "flowfield_kernel.inc"

void CopyCurrentMacroVar() {
    for (auto& pair : g_MacroVars()) {
        const int varId{pair.first};
        RealField& macroVar{pair.second};
        RealField& macroVarCopy{g_MacroVarsCopy().at(varId)};
        for (const auto& idBlock : g_Block()) {
            const Block& block{idBlock.second};
            std::vector<int> iterRng;
            iterRng.assign(block.WholeRange().begin(),
                           block.WholeRange().end());
            const int blockIdx{block.ID()};
            ops_par_loop(KerCopyMacroVars, "KerCopyMacroVars", block.Get(),
                         SpaceDim(), iterRng.data(),
                         ops_arg_dat(macroVar.at(blockIdx), 1, LOCALSTENCIL,
                                     "double", OPS_READ),
                         ops_arg_dat(macroVarCopy.at(blockIdx), 1, LOCALSTENCIL,
                                     "double", OPS_RW));
        }
    }
}

void CalcResidualError3D() {
    std::map<int, Real> diff;
    for (const auto& pair : g_MacroVars()) {
        const int varId{pair.first};
        const RealField& macroVar{pair.second};
        const RealField& macroVarCopy{g_MacroVarsCopy().at(varId)};
        Real error{0};
        diff.emplace(varId, error);
        for (const auto& idBlock : g_Block()) {
            const Block& block{idBlock.second};
            std::vector<int> iterRng;
            iterRng.assign(block.WholeRange().begin(),
                           block.WholeRange().end());
            const int blockIdx{block.ID()};
            ops_par_loop(KerCalcMacroVarSquareofDifference,
                         "KerCalcMacroVarSquareofDifference", block.Get(),
                         SpaceDim(), iterRng.data(),
                         ops_arg_dat(macroVar.at(blockIdx), 1, LOCALSTENCIL,
                                     "double", OPS_READ),
                         ops_arg_dat(macroVarCopy.at(blockIdx), 1, LOCALSTENCIL,
                                     "double", OPS_READ),
                         // TODO if we can change "double" here?
                         ops_arg_reduce(g_ResidualErrorHandle().at(varId), 1,
                                        "double", OPS_INC));
        }
    }
    // TODO:check if ops_reduction_results works directly for multi-block
    for (const auto& pair : g_MacroVars()) {
        ops_reduction_result(g_ResidualErrorHandle().at(pair.first),
                             &diff.at(pair.first));
    }

    CopyCurrentMacroVar();

    for (const auto& pair : g_MacroVars()) {
        const int varId{pair.first};
        const RealField& macroVar{pair.second};
        for (const auto& idBlock : g_Block()) {
            const Block& block{idBlock.second};
            std::vector<int> iterRng;
            iterRng.assign(block.WholeRange().begin(),
                           block.WholeRange().end());
            const int blockIdx{block.ID()};
            ops_par_loop(KerCalcMacroVarSquare, "KerCalcMacroVarSquare3D",
                         block.Get(), SpaceDim(), iterRng.data(),
                         ops_arg_dat(macroVar.at(blockIdx), 1, LOCALSTENCIL,
                                     "double", OPS_READ),
                         ops_arg_reduce(g_ResidualErrorHandle().at(varId), 1,
                                        "double", OPS_INC));
        }
    }

    for (const auto& pair : g_MacroVars()) {
        int varId{pair.first};
        Real sum{0};
        ops_reduction_result(g_ResidualErrorHandle().at(varId), &sum);
        g_ResidualError().at(varId) = diff.at(varId) / sum;
    }
}

void CopyDistribution3D(RealField& fDest, RealField& fSrc) {
    for (const auto& idBlock : g_Block()) {
        const Block& block{idBlock.second};
        std::vector<int> iterRng;
        iterRng.assign(block.WholeRange().begin(), block.WholeRange().end());
        const int blockIndex{block.ID()};
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));
    }
}

// This routine is necessary now due to the following reason:
// 1. the collision process might not be implemented at some kind of boundary
// points so that f_stage will not be updated.
// 2. The periodic boundary is acccutally implemented in the stream process now,
// which needs the information at halo points.
// The routine shall be removed if the stream process can be implemented in a
// way that f_stage is not necessary.
void CopyBlockEnvelopDistribution3D(Field<Real>& fDest, Field<Real>& fSrc) {
    // int haloIterRng[]{0, 0, 0, 0, 0, 0};
    for (const auto& idBlock : g_Block()) {
        const Block& block{idBlock.second};
        std::vector<int> iterRng;
        iterRng.assign(block.IminRange().begin(), block.IminRange().end());
        const int blockIndex{block.ID()};
        // haloIterRng[0] = iterRng.data()[0] - 1;
        // haloIterRng[1] = iterRng.data()[1];
        // haloIterRng[2] = iterRng.data()[2] - 1;
        // haloIterRng[3] = iterRng.data()[3] + 1;
        // haloIterRng[4] = iterRng.data()[4] - 1;
        // haloIterRng[5] = iterRng.data()[5] + 1;
        // ops_printf("IterRngImin= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));

        iterRng.assign(block.ImaxRange().begin(), block.ImaxRange().end());
        // haloIterRng[0] = iterRng.data()[0];
        // haloIterRng[1] = iterRng.data()[1] + 1;
        // haloIterRng[2] = iterRng.data()[2] - 1;
        // haloIterRng[3] = iterRng.data()[3] + 1;
        // haloIterRng[4] = iterRng.data()[4] - 1;
        // haloIterRng[5] = iterRng.data()[5] + 1;
        // ops_printf("IterRngImax= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));

        iterRng.assign(block.JminRange().begin(), block.JminRange().end());
        // haloIterRng[0] = iterRng.data()[0] - 1;
        // haloIterRng[1] = iterRng.data()[1] + 1;
        // haloIterRng[2] = iterRng.data()[2] - 1;
        // haloIterRng[3] = iterRng.data()[3];
        // haloIterRng[4] = iterRng.data()[4] - 1;
        // haloIterRng[5] = iterRng.data()[5] + 1;
        // ops_printf("IterRngJmin= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));
        iterRng.assign(block.JmaxRange().begin(), block.JmaxRange().end());
        // haloIterRng[0] = iterRng.data()[0] - 1;
        // haloIterRng[1] = iterRng.data()[1] + 1;
        // haloIterRng[2] = iterRng.data()[2];
        // haloIterRng[3] = iterRng.data()[3] + 1;
        // haloIterRng[4] = iterRng.data()[4] - 1;
        // haloIterRng[5] = iterRng.data()[5] + 1;
        // ops_printf("IterRngJmax= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));
        iterRng.assign(block.KminRange().begin(), block.KminRange().end());
        // haloIterRng[0] = iterRng.data()[0] - 1;
        // haloIterRng[1] = iterRng.data()[1] + 1;
        // haloIterRng[2] = iterRng.data()[2] - 1;
        // haloIterRng[3] = iterRng.data()[3] + 1;
        // haloIterRng[4] = iterRng.data()[4] - 1;
        // haloIterRng[5] = iterRng.data()[5];
        // ops_printf("IterRngKmin= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));
        iterRng.assign(block.KmaxRange().begin(), block.KmaxRange().end());
        // haloIterRng[0] = iterRng.data()[0] - 1;
        // haloIterRng[1] = iterRng.data()[1] + 1;
        // haloIterRng[2] = iterRng.data()[2] - 1;
        // haloIterRng[3] = iterRng.data()[3] + 1;
        // haloIterRng[4] = iterRng.data()[4];
        // haloIterRng[5] = iterRng.data()[5] + 1;
        // ops_printf("IterRngKmax= %d %d %d %d %d %d\n", haloIterRng[0],
        //            haloIterRng[1], haloIterRng[2], haloIterRng[3],
        //            haloIterRng[4], haloIterRng[5]);
        ops_par_loop(KerCopyf, "KerCopyf", block.Get(), SpaceDim(),
                     iterRng.data(),
                     ops_arg_dat(fDest[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_WRITE),
                     ops_arg_dat(fSrc[blockIndex], NUMXI, LOCALSTENCIL,
                                 "double", OPS_READ));
    }
}

void NormaliseF3D(Real* ratio) {
    for (auto idBlock : g_Block()) {
        Block& block{idBlock.second};
        std::vector<int> iterRng;
        iterRng.assign(block.WholeRange().begin(), block.WholeRange().end());
        const int blockIdx{block.ID()};
        ops_par_loop(KerNormaliseF, "KerNormaliseF", block.Get(), SpaceDim(),
                     iterRng.data(), ops_arg_gbl(ratio, 1, "double", OPS_READ),
                     ops_arg_dat(g_f()[blockIdx], NUMXI, LOCALSTENCIL, "double",
                                 OPS_RW));
    }
}

void AssignCoordinates(const Block& block,
                       const std::vector<std::vector<Real>>& blockCoordinates) {
#ifdef OPS_2D
    if (SpaceDim() == 2) {
        std::vector<int> range(2 * SpaceDim());
        range.assign(block.WholeRange().begin(), block.WholeRange().end());
        const Real* coordinateX{blockCoordinates.at(0).data()};
        const int sizeX{block.Size().at(0)};
        const Real* coordinateY{blockCoordinates.at(1).data()};
        const int sizeY{block.Size().at(1)};
        ops_par_loop(KerSetCoordinates, "KerSetCoordinates",
                    block.Get(), SpaceDim(), range.data(),
                     ops_arg_dat(g_CoordinateXYZ()[blockIndex], SpaceDim(),
                                 LOCALSTENCIL, "double", OPS_WRITE),
                     ops_arg_idx(),
                     ops_arg_gbl(coordinateX, sizeX, "double", OPS_READ),
                     ops_arg_gbl(coordinateY, sizeY, "double", OPS_READ));
    }
#endif

#ifdef OPS_3D
    if (SpaceDim() == 3) {
        std::vector<int> range(2 * SpaceDim());
        range.assign(block.WholeRange().begin(), block.WholeRange().end());
        const Real* coordinateX{blockCoordinates.at(0).data()};
        const int sizeX{block.Size().at(0)};
        const Real* coordinateY{blockCoordinates.at(1).data()};
        const int sizeY{block.Size().at(1)};
        const Real* coordinateZ{blockCoordinates.at(2).data()};
        const int sizeZ{block.Size().at(2)};
        ops_par_loop(KerSetCoordinates3D, "KerSetCoordinates3D", block.Get(),
                     SpaceDim(), range.data(),
                     ops_arg_dat(g_CoordinateXYZ()[block.ID()], SpaceDim(),
                                 LOCALSTENCIL, "double", OPS_WRITE),
                     ops_arg_idx(),
                     ops_arg_gbl(coordinateX, sizeX, "double", OPS_READ),
                     ops_arg_gbl(coordinateY, sizeY, "double", OPS_READ),
                     ops_arg_gbl(coordinateZ, sizeZ, "double", OPS_READ)

        );
    }
#endif
}

void  SetBlockGeometryProperty(const Block& block) {
    int geometryProperty = (int)VG_Fluid;
    // int* iterRange = BlockIterRng(blockIndex, IterRngBulk());
    std::vector<int> iterRange;
    iterRange.assign(block.BulkRange().begin(), block.BulkRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(),
                 SpaceDim(), iterRange.data(),
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    // specify halo points
    geometryProperty = VG_ImmersedSolid;
    iterRange.assign(block.JminRange().begin(), block.JminRange().end());
    int* haloIterRng = new int[2 * SpaceDim()];
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] - 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    ops_par_loop(KerSetIntField, "KerSetIntField",
                 block.Get(), SpaceDim(), haloIterRng,
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    iterRange.assign(block.JmaxRange().begin(), block.JmaxRange().end());
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] + 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    ops_par_loop(KerSetIntField, "KerSetIntField",
                 block.Get(), SpaceDim(), haloIterRng,
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    iterRange.assign(block.IminRange().begin(), block.IminRange().end());
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] - 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), haloIterRng,
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    iterRange.assign(block.ImaxRange().begin(), block.ImaxRange().end());
    haloIterRng[0] = iterRange[0] + 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), haloIterRng,
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    if (3 == SpaceDim()) {
        iterRange.assign(block.KminRange().begin(), block.KminRange().end());
        haloIterRng[0] = iterRange[0] - 1;
        haloIterRng[1] = iterRange[1] + 1;
        haloIterRng[2] = iterRange[2] - 1;
        haloIterRng[3] = iterRange[3] + 1;
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] - 1;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), haloIterRng,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        iterRange.assign(block.KmaxRange().begin(), block.KmaxRange().end());
        haloIterRng[0] = iterRange[0] - 1;
        haloIterRng[1] = iterRange[1] + 1;
        haloIterRng[2] = iterRange[2] - 1;
        haloIterRng[3] = iterRange[3] + 1;
        haloIterRng[4] = iterRange[4] + 1;
        haloIterRng[5] = iterRange[5] + 1;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), haloIterRng,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
    }

    // specify domain
    geometryProperty = VG_JP;
    iterRange.assign(block.JminRange().begin(), block.JminRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), iterRange.data(),
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    geometryProperty = VG_JM;
    iterRange.assign(block.JmaxRange().begin(), block.JmaxRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), iterRange.data(),
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    geometryProperty = VG_IP;
    iterRange.assign(block.IminRange().begin(), block.IminRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), iterRange.data(),
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    geometryProperty = VG_IM;
    iterRange.assign(block.ImaxRange().begin(), block.ImaxRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField",
                block.Get(), SpaceDim(), iterRange.data(),
                 ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                 ops_arg_dat(g_GeometryProperty()[block.ID()], 1, LOCALSTENCIL,
                             "int", OPS_WRITE));
    if (3 == SpaceDim()) {
        geometryProperty = VG_KP;
        iterRange.assign(block.KminRange().begin(), block.KminRange().end());
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iterRange.data(),
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        geometryProperty = VG_KM;
        iterRange.assign(block.KmaxRange().begin(), block.KmaxRange().end());
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iterRange.data(),
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
    }

    const int nx{(int)block.Size().at(0)};
    const int ny{(int)block.Size().at(1)};
    // 2D Domain corner points four types
    if (2 == SpaceDim()) {
        int iminjmin[]{0, 1, 0, 1};
        geometryProperty = VG_IPJP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminjmax[] = {0, 1, ny - 1, ny};
        geometryProperty = VG_IPJM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmax[] = {nx - 1, nx, ny - 1, ny};
        geometryProperty = VG_IMJM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmin[] = {nx - 1, nx, 0, 1};
        geometryProperty = VG_IMJP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
    }

    if (3 == SpaceDim()) {
        const int nz{(int)block.Size().at(2)};
        // 3D Domain edges 12 types
        int iminjmin[]{0, 1, 0, 1, 0, nz};
        geometryProperty = VG_IPJP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminjmax[]{0, 1, ny - 1, ny, 0, nz};
        geometryProperty = VG_IPJM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmax[]{nx - 1, nx, ny - 1, ny, 0, nz};
        geometryProperty = VG_IMJM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmin[]{nx - 1, nx, 0, 1, 0, nz};
        geometryProperty = VG_IMJP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));

        int iminkmin[]{0, 1, 0, ny, 0, 1};
        geometryProperty = VG_IPKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminkmax[]{0, 1, 0, ny, nz - 1, nz};
        geometryProperty = VG_IPKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxkmax[]{nx - 1, nx, 0, ny, nz - 1, nz};
        geometryProperty = VG_IMKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxkmin[]{nx - 1, nx, 0, ny, 0, 1};
        geometryProperty = VG_IMKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));

        int jminkmin[]{0, nx, 0, 1, 0, 1};
        geometryProperty = VG_JPKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), jminkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int jminkmax[]{0, nx, 0, 1, nz - 1, nz};
        geometryProperty = VG_JPKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), jminkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int jmaxkmax[]{0, nx, ny - 1, ny, nz - 1, nz};
        geometryProperty = VG_JMKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), jmaxkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int jmaxkmin[]{0, nx, ny - 1, ny, 0, 1};
        geometryProperty = VG_JMKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), jmaxkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));

        // 3D domain corners 8 types
        int iminjminkmin[]{0, 1, 0, 1, 0, 1};
        geometryProperty = VG_IPJPKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjminkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminjminkmax[]{0, 1, 0, 1, nz - 1, nz};
        geometryProperty = VG_IPJPKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjminkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminjmaxkmin[]{0, 1, ny - 1, ny, 0, 1};
        geometryProperty = VG_IPJMKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmaxkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int iminjmaxkmax[]{0, 1, ny - 1, ny, nz - 1, nz};
        geometryProperty = VG_IPJMKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), iminjmaxkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjminkmin[]{nx - 1, nx, 0, 1, 0, 1};
        geometryProperty = VG_IMJPKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjminkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjminkmax[]{nx - 1, nx, 0, 1, nz - 1, nz};
        geometryProperty = VG_IMJPKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjminkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmaxkmin[]{nx - 1, nx, ny - 1, ny, 0, 1};
        geometryProperty = VG_IMJMKP_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmaxkmin,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
        int imaxjmaxkmax[]{nx - 1, nx, ny - 1, ny, nz - 1, nz};
        geometryProperty = VG_IMJMKM_I;
        ops_par_loop(KerSetIntField, "KerSetIntField",
                    block.Get(), SpaceDim(), imaxjmaxkmax,
                     ops_arg_gbl(&geometryProperty, 1, "int", OPS_READ),
                     ops_arg_dat(g_GeometryProperty()[block.ID()], 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
    }
}

void SetBoundaryNodeType() {
    for (auto& boundary : BlockBoundaries()) {
        const int boundaryType{(int)boundary.boundaryType};
        const Block& block{g_Block().at(boundary.blockIndex)};
        std::vector<int> iterRange{
            BoundarySurfaceRange(block, boundary.boundarySurface)};
        const int compoId{boundary.componentID};
        // Specify general boundary type
        ops_par_loop(
            KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
            iterRange.data(), ops_arg_gbl(&boundaryType, 1, "int", OPS_READ),
            ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                        LOCALSTENCIL, "int", OPS_WRITE));
    }
}

void SetBulkandHaloNodesType(const Block& block , int compoId) {
    const int fluidType{(int)VertexType::Fluid};
    const int immersedSolidType{(int)VertexType::ImmersedSolid};
    std::vector<int> iterRange;
    iterRange.assign(block.BulkRange().begin(), block.BulkRange().end());
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                 iterRange.data(), ops_arg_gbl(&fluidType, 1, "int", OPS_READ),
                 ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                             LOCALSTENCIL, "int", OPS_WRITE));

    iterRange.assign(block.JminRange().begin(), block.JminRange().end());
    // Specify halo points
    int* haloIterRng = new int[2 * SpaceDim()];
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] - 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                 haloIterRng,
                 ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                 ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                             LOCALSTENCIL, "int", OPS_WRITE));

    iterRange.assign(block.JmaxRange().begin(), block.JmaxRange().end());
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] + 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    // Specify halo points
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                 haloIterRng,
                 ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                 ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                             LOCALSTENCIL, "int", OPS_WRITE));

    iterRange.assign(block.IminRange().begin(), block.IminRange().end());
    haloIterRng[0] = iterRange[0] - 1;
    haloIterRng[1] = iterRange[1] - 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    // Specify halo points
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                 haloIterRng,
                 ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                 ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                             LOCALSTENCIL, "int", OPS_WRITE));

    iterRange.assign(block.ImaxRange().begin(), block.ImaxRange().end());
    haloIterRng[0] = iterRange[0] + 1;
    haloIterRng[1] = iterRange[1] + 1;
    haloIterRng[2] = iterRange[2] - 1;
    haloIterRng[3] = iterRange[3] + 1;
    if (3 == SpaceDim()) {
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] + 1;
    }
    // Specify halo points
    ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                 haloIterRng,
                 ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                 ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                             LOCALSTENCIL, "int", OPS_WRITE));

    if (3 == SpaceDim()) {
        iterRange.assign(block.KminRange().begin(), block.KminRange().end());
        haloIterRng[0] = iterRange[0] - 1;
        haloIterRng[1] = iterRange[1] + 1;
        haloIterRng[2] = iterRange[2] - 1;
        haloIterRng[3] = iterRange[3] + 1;
        haloIterRng[4] = iterRange[4] - 1;
        haloIterRng[5] = iterRange[5] - 1;
        // Specify halo points
        ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                     haloIterRng,
                     ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                     ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));

        iterRange.assign(block.KmaxRange().begin(), block.KmaxRange().end());
        haloIterRng[0] = iterRange[0] - 1;
        haloIterRng[1] = iterRange[1] + 1;
        haloIterRng[2] = iterRange[2] - 1;
        haloIterRng[3] = iterRange[3] + 1;
        haloIterRng[4] = iterRange[4] + 1;
        haloIterRng[5] = iterRange[5] + 1;
        // Specify halo points
        ops_par_loop(KerSetIntField, "KerSetIntField", block.Get(), SpaceDim(),
                     haloIterRng,
                     ops_arg_gbl(&immersedSolidType, 1, "int", OPS_READ),
                     ops_arg_dat(g_NodeType().at(compoId).at(block.ID()), 1,
                                 LOCALSTENCIL, "int", OPS_WRITE));
    }
    FreeArrayMemory(haloIterRng);
}

