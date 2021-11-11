/**
 * Copyright 2019 United Kingdom Research and Innovation
 *
 * Authors: See AUTHORS
 *
 * Contact: [jianping.meng@stfc.ac.uk and/or jpmeng@gmail.com]
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice
 *    this list of conditions and the following disclaimer in the documentation
 *    and or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * ANDANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
*/
#include <random>
std::random_device rd;
std::default_random_engine e(rd());
std::uniform_real_distribution<double> kk(0.2509,0.2609);
#ifndef CAVITY3D_KERNEL_INC
#define CAVITY3D_KERNEL_INC


void KerSetInitialMacroVars(ACC<Real>& rho, ACC<Real>& u, ACC<Real>& v,
                            ACC<Real>& w, const ACC<Real>& coordinates,
                            const int* idx) {
    rho(0, 0, 0) = kk(e);
    u(0, 0, 0) = 0;
    v(0, 0, 0) = 0;
    w(0, 0, 0) = 0;
}
void KerUpdateBodyForce(ACC<Real>& acceration, const ACC<Real>& Rho, const ACC<Real>& P,
                        const ACC<Real>& Psi, const ACC<int>& nodeType, const int* lattIdx) {
                VertexType vt = (VertexType)nodeType(0, 0, 0);
                if (vt != VertexType::ImmersedSolid) {
                    Real G = -1.0;
                    Real Fx = 0;
                    Real Fy = 0;
                    Real Fz = 0;
                    if (P(0, 0, 0) - Rho(0, 0, 0) / CS / CS > 0) G = 1.0;
                    for (int xiIndex = lattIdx[0]; xiIndex <= lattIdx[1]; xiIndex++) {
                         Real cx = XI[xiIndex * LATTDIM];
                         Real cy = XI[xiIndex * LATTDIM + 1];
                         Real cz = XI[xiIndex * LATTDIM + 2]; 
                         Fx += cx * Psi(cx, cy, cz) * WEIGHTS[xiIndex];
                         Fy += cy * Psi(cx, cy, cz) * WEIGHTS[xiIndex];
                         Fz += cz * Psi(cx, cy, cz) * WEIGHTS[xiIndex];
                    }
                         Fx *= -G * Psi(0, 0, 0);
                         Fy *= -G * Psi(0, 0, 0);
                         Fz *= -G * Psi(0, 0, 0);
                         acceration(0,0,0,0) = Fx;
                         acceration(1,0,0,0) = Fy;
                         acceration(2,0,0,0) = Fz;
                        }
}
#endif // CAVITY3D_KERNEL_INC
