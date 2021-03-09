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

/*! @brief   Implementing functions related to the flow field
 * @author  Jianping Meng
 * @details Implementing functions related to create the flow
 * field (allocate memory), set up the geometry and the boundary
 * property, and deallocate the memory.
 */

#ifndef BLOCK_H
#define BLOCK_H
#include <map>
#include <string>
#include <vector>
#include <cassert>

#include "ops_seq_v2.h"
#include "type.h"
class Block {
   private:
    ops_block block;
#ifdef OPS_3D
    const int spaceDim{3};
#endif
#ifdef OPS_2D
    const int spaceDim{2};
#endif
    std::string name;
    SizeType id;
    std::vector<int> size;
    std::vector<int> wholeRange;
    std::vector<int> bulkRange;
    std::vector<int> iminRange;
    std::vector<int> imaxRange;
    std::vector<int> jminRange;
    std::vector<int> jmaxRange;
#ifdef OPS_3D
    std::vector<int> kminRange;
    std::vector<int> kmaxRange;
#endif
   public:
    Block(const SizeType blockId, const std::string& blockName,
          const std::vector<int> & blockSize);
    SizeType ID() const { return id; };
    const std::string& Name() const { return name; };
    const ops_block& Get() const { return block; };
    ops_block& Get() { return block; };
    const std::vector<int>& Size() const { return size; };
    const int* pSize() const { return size.data(); };
    const std::vector<int>& WholeRange() const { return wholeRange; };
    const std::vector<int>& BulkRange() const { return bulkRange; };
    const std::vector<int>& ImaxRange() const { return imaxRange; };
    const std::vector<int>& IminRange() const { return iminRange; };
    const std::vector<int>& JmaxRange() const { return jmaxRange; };
    const std::vector<int>& JminRange() const { return jminRange; };
#ifdef OPS_3D
    const std::vector<int>& KmaxRange() const { return kmaxRange; };
    const std::vector<int>& KminRange() const { return kminRange; };
#endif
};

typedef std::map<SizeType, Block> BlockGroup;
#endif  // BLOCK_H