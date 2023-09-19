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
 * @brief  Ryoanji multipole and related types
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"

namespace ryoanji
{

template<class T>
using Vec3 = cstone::Vec3<T>;

template<class T>
using Vec4 = cstone::Vec4<T>;

using TreeNodeIndex = cstone::TreeNodeIndex;
using LocalIndex    = cstone::LocalIndex;

template<size_t P>
struct TermSize : public stl::integral_constant<size_t, P*(P + 1) * (P + 2) / 6>
{
};

template<class T, size_t P>
using SphericalMultipole = util::array<T, TermSize<P>{}>;

template<class MType>
struct IsSpherical
    : public stl::integral_constant<size_t, MType{}.size() == TermSize<2>{} || MType{}.size() == TermSize<4>{}>
{
};

template<int ArraySize, class = void>
struct ExpansionOrder
{
};

template<>
struct ExpansionOrder<TermSize<1>{}> : stl::integral_constant<size_t, 1>
{
};

template<>
struct ExpansionOrder<TermSize<2>{}> : stl::integral_constant<size_t, 2>
{
};

template<>
struct ExpansionOrder<TermSize<3>{}> : stl::integral_constant<size_t, 3>
{
};

template<>
struct ExpansionOrder<TermSize<4>{}> : stl::integral_constant<size_t, 4>
{
};

} // namespace ryoanji
