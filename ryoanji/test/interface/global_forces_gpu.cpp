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
 * @brief Compute an octree, multipoles and forces on GPUs from a set of particles distributed across ranks
 *        and compare against a single-node reference computed from the same set.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"

using namespace ryoanji;

template<class T, class KeyType>
static int multipoleHolderTest(int thisRank, int numRanks)
{
    using MultipoleType              = CartesianQuadrupole<T>;
    const LocalIndex numParticles    = (100000 / numRanks) * numRanks;
    unsigned         bucketSize      = numParticles / (100 * numRanks);
    unsigned         bucketSizeLocal = std::min(64u, bucketSize);
    float            theta           = 0.5;
    T                G               = 1.0;

    cstone::Box<T> box{-1, 1};
    // setting numShells to non-zero won't match the direct sum reference because it only includes the replica shells,
    // but not the Ewald corrections, whereas the Barnes-Hut implementation contains both
    int numShells = 0;

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box);

    std::vector<T> globalH(numParticles, 0.1);
    adjustSmoothingLength<KeyType>(globalH.size(), 5, 10, coords.x(), coords.y(), coords.z(), globalH, box);

    std::vector<T> globalMasses(numParticles, 1.0 / numParticles);

    LocalIndex firstIndex = (numParticles * thisRank) / numRanks;
    LocalIndex lastIndex  = (numParticles * (thisRank + 1)) / numRanks;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T>       x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T>       y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T>       z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T>       h(globalH.begin() + firstIndex, globalH.begin() + lastIndex);
    std::vector<T>       m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);
    std::vector<KeyType> h_keys(x.size());

    cstone::Domain<KeyType, T, cstone::GpuTag> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    thrust::device_vector<KeyType> d_keys = h_keys;
    thrust::device_vector<T>       d_x = x, d_y = y, d_z = z, d_h = h;
    thrust::device_vector<T>       d_m = m;
    thrust::device_vector<T>       s1;
    thrust::device_vector<T>       s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    h_keys.resize(domain.nParticles());
    thrust::copy(d_keys.begin() + domain.startIndex(), d_keys.begin() + domain.endIndex(), h_keys.begin());

    /*! The range [firstGlobalIdx:lastGlobalIdx] of the global set @a coords is identical to the locally
     *  present particles contained in the range [domain.startIndex():domain.endIndex()] of arrays (d_x, d_y, d_z)
     *  This is true because @a coords is SFC sorted, therefore after domain.syncGrav, each rank will have
     *  one continuous section of the global @a coords set, which will largely overlap with the initial pre-sync range
     *  [firstIndex:lastIndex] but might be shifted left or right by a small distance.
     */
    LocalIndex firstGlobalIdx =
        std::lower_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.front()) -
        coords.particleKeys().begin();
    LocalIndex lastGlobalIdx =
        std::upper_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.back()) -
        coords.particleKeys().begin();

    //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    //! the focused octree, structure only
    auto                                         octree  = focusTree.octreeViewAcc();
    gsl::span<const cstone::SourceCenterType<T>> centers = focusTree.expansionCentersAcc();

    std::vector<MultipoleType> multipoles(octree.numNodes);
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data(), multipoles.data());

    // Check Barnes-Hut accelerations from distributed particle set
    // against the direct-sum reference computed with the (single node) common pool
    bool                pass;
    std::vector<double> firstPercentiles(numRanks), maxErrors(numRanks);
    {
        thrust::device_vector<T> d_ax, d_ay, d_az;
        d_ax = std::vector<T>(domain.nParticlesWithHalos(), 0);
        d_ay = std::vector<T>(domain.nParticlesWithHalos(), 0);
        d_az = std::vector<T>(domain.nParticlesWithHalos(), 0);

        auto   grp         = multipoleHolder.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d_x),
                                                                  rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), domain.focusTree(),
                                                                  domain.layout().data(), domain.box());
        double bhPotential = multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                                     rawPtr(d_h), G, 0, box, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az));

        // download BH accelerations for locally present particles
        thrust::host_vector<T> ax(d_ax.begin() + domain.startIndex(), d_ax.begin() + domain.endIndex());
        thrust::host_vector<T> ay(d_ay.begin() + domain.startIndex(), d_ay.begin() + domain.endIndex());
        thrust::host_vector<T> az(d_az.begin() + domain.startIndex(), d_az.begin() + domain.endIndex());

        thrust::device_vector<T> d_xref = coords.x(), d_yref = coords.y(), d_zref = coords.z();
        thrust::device_vector<T> d_mref = globalMasses;
        thrust::device_vector<T> d_href = globalH;

        thrust::device_vector<T> d_potref, d_axref, d_ayref, d_azref;
        reallocateDevice(d_potref, numParticles, 1.0);
        reallocateDevice(d_axref, numParticles, 1.0);
        reallocateDevice(d_ayref, numParticles, 1.0);
        reallocateDevice(d_azref, numParticles, 1.0);

        // reference direct sum calculation with the global set of sources
        directSum(firstGlobalIdx, lastGlobalIdx, numParticles, {box.lx(), box.ly(), box.lz()}, numShells,
                  rawPtr(d_xref), rawPtr(d_yref), rawPtr(d_zref), rawPtr(d_mref), rawPtr(d_href), rawPtr(d_potref),
                  rawPtr(d_axref), rawPtr(d_ayref), rawPtr(d_azref));

        // download reference direct sum accelerations due to global source set
        thrust::host_vector<T> pRef(d_potref.begin() + firstGlobalIdx, d_potref.begin() + lastGlobalIdx);
        thrust::host_vector<T> axRef(d_axref.begin() + firstGlobalIdx, d_axref.begin() + lastGlobalIdx);
        thrust::host_vector<T> ayRef(d_ayref.begin() + firstGlobalIdx, d_ayref.begin() + lastGlobalIdx);
        thrust::host_vector<T> azRef(d_azref.begin() + firstGlobalIdx, d_azref.begin() + lastGlobalIdx);

        double         potentialSumRef = 0;
        std::vector<T> errors(ax.size());
        for (int i = 0; i < ax.size(); i++)
        {
            potentialSumRef += pRef[i];
            Vec3<T> ref   = {axRef[i], ayRef[i], azRef[i]};
            Vec3<T> probe = {ax[i], ay[i], az[i]};
            errors[i]     = std::sqrt(norm2(ref - probe) / norm2(ref));
        }
        potentialSumRef *= 0.5 * G;
        std::sort(begin(errors), end(errors));

        double err1pc = errors[errors.size() * 0.99];
        double errmax = errors.back();

        MPI_Allgather(&err1pc, 1, MPI_DOUBLE, firstPercentiles.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(&errmax, 1, MPI_DOUBLE, maxErrors.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

        double bhPotentialGlob, potentialSumRefGlob;
        MPI_Allreduce(&bhPotential, &bhPotentialGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&potentialSumRef, &potentialSumRefGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double atol1pc = 1e-3;
        double atolmax = 3e-2;
        double ptol    = 1e-2;

        if (thisRank == 0)
        {
            for (int i = 0; i < numRanks; ++i)
                std::cout << "rank " << i << " 1st-percentile acc error " << firstPercentiles[i] << ", max acc error "
                          << maxErrors[i] << std::endl;

            std::cout << "global reference potential " << potentialSumRefGlob << ", BH global potential "
                      << bhPotentialGlob << std::endl;
        }

        bool passAcc1pc = *std::max_element(firstPercentiles.begin(), firstPercentiles.end()) < atol1pc;
        bool passAccMax = *std::max_element(maxErrors.begin(), maxErrors.end()) < atolmax;
        bool passPot    = std::abs((bhPotentialGlob - potentialSumRefGlob) / potentialSumRefGlob) < ptol;

        pass = passAcc1pc && passAccMax && passPot;
    }

    if (thisRank == 0)
    {
        std::string testResult = pass ? "PASS" : "FAIL";
        std::cout << "Test result: " << testResult << std::endl;
    }

    if (pass) { return EXIT_SUCCESS; }
    else { return EXIT_FAILURE; }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = multipoleHolderTest<float, uint64_t>(rank, numRanks);

    MPI_Finalize();

    return testResult;
}
