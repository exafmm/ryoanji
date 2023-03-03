/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Interface for calculation of multipole moments
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>

#include "cstone/focus/octree_focus_mpi.hpp"
#include "ryoanji/nbody/types.h"

namespace ryoanji
{

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
class MultipoleHolder
{
public:
    MultipoleHolder();

    ~MultipoleHolder();

    /*! @brief Compute multipoles for given particles and octree, performs MPI communication
     *
     * @param[in]  x              particle x coordinates, on device
     * @param[in]  y              particle y coordinates, on device
     * @param[in]  z              particle z coordinates, on device
     * @param[in]  m              particle masses, on device
     * @param[in]  globalOctree   fully linked octree used for domain decomposition on CPU
     * @param[in]  focusTree      fully linked octree focused on local domain (but still global) on device
     * @param[in]  layout         index offsets per tree leaf cell into particle buffers, on device
     * @param[out] multipoles     multipole temp storage, on CPU
     *
     * Particle buffers need to be SFC-sorted and indexed to match @p layout.
     */
    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Octree<KeyType>& globalOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const cstone::LocalIndex* layout,
                 MType* multipoles);

    /*! @brief Compute gravitational accelerations for all particles in the range [firstBody:lastBody]
     *
     * @param[in] firstBody  first index in particles buffers to compute accelerations for
     * @param[in] lastBody   last index (exclusive) in particles buffers to compute accelerations for
     * @param[in] x         particle x coordinates, on device
     * @param[in] y         particle y coordinates, on device
     * @param[in] z         particle z coordinates, on device
     * @param[in] m         particle masses
     * @param[in] h         particle smoothing lengths, on device
     * @param[in] G         gravitational constant
     * @param[out] ax       output accelerations, on device
     * @param[out] ay
     * @param[out] az
     * @return
     */
    float compute(LocalIndex firstBody, LocalIndex lastBody, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                  const Th* h, Tc G, Ta* ax, Ta* ay, Ta* az);

    //! @brief return a tuple with sumP2P, maxP2P, sumM2P, maxM2P from the last call to compute
    util::array<uint64_t, 4> readStats() const;

    const MType* deviceMultipoles() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ryoanji
