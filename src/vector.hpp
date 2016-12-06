/**
 * @file vector.hpp
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * any later version. Please see https://gnu.org/licenses/gpl.html
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
**/

#pragma once
// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <random>
#include <ratio>

// Internal headers
#include "common.hpp"

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Simple vector ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Simple vector class.
 * @param dim Vector dimension
**/
template<nat_t dim> class Vector final {
    static_assert(dim > 0, "Invalid vector dimension");
private:
    val_t vec[dim]; // Vector
public:
    /** Uninitialized constructor.
    **/
    Vector() {}
    /** Copy constructor.
     * @param copy Vector to copy
    **/
    Vector(Vector<dim> const& copy) {
        for (nat_t i = 0; i < dim; i++)
            vec[i] = copy.get(i);
    }
    /** Copy constructor.
     * @param copy Scalar to copy
    **/
    Vector(val_t copy) {
        for (nat_t i = 0; i < dim; i++)
            vec[i] = copy;
    }
    /** Initializer list constructor.
     * @param params Initial values (cardinality must be the dimension of the vector)
    **/
    Vector(::std::initializer_list<val_t> params) {
        /// FIXME: Non-constexpr condition here. Missing a 'correction' brought by C++17 ? See 'http://en.cppreference.com/w/cpp/utility/initializer_list/size' then 'http://en.cppreference.com/w/cpp/iterator/distance'.
        // static_assert(params.size() != dim, "Wrong initializer list size");
        nat_t i = 0;
        for (val_t v: params)
            vec[i++ % dim] = v; /// NOTE: Modulo to avoid undefined behavior
    }
public:
    /** Get a single coordinate.
     * @param id Coordinate id
     * @return Coordinate value
    **/
    val_t get(nat_t id) const {
        return vec[id];
    }
    /** Set a single coordinate.
     * @param id Coordinate id
     * @param cv Coordinate value
    **/
    void set(nat_t id, val_t cv) {
        vec[id] = cv;
    }
public:
    /** Copy assignment.
     * @param x Vector to copy
     * @return Current vector
    **/
    Vector<dim>& operator=(Vector<dim> const& x) {
        for (nat_t i = 0; i < dim; i++)
            set(i, x.get(i));
        return *this;
    }
    /** Copy assignment.
     * @param x Scalar to copy
     * @return Current vector
    **/
    Vector<dim>& operator=(val_t x) {
        for (nat_t i = 0; i < dim; i++)
            set(i, x);
        return *this;
    }
    /** Initializer list assignment.
     * @param params Initial values (cardinality must be the dimension of the vector)
     * @return Current vector
    **/
    Vector<dim>& operator=(::std::initializer_list<val_t> params) {
        /// FIXME: Non-constexpr condition here. Same remark.
        // static_assert(params.size() != dim, "Wrong initializer list size");
        nat_t i = 0;
        for (val_t v: params)
            vec[i++ % dim] = v; /// NOTE: Modulo to avoid undefined behavior
        return *this;
    }
    /** Vector difference.
     * @param x Vector to substract
     * @return Current vector
    **/
    Vector<dim>& operator-=(Vector<dim> const& x) {
        for (nat_t i = 0; i < dim; i++)
            set(i, get(i) - x.get(i));
        return *this;
    }
    /** Scalar product.
     * @param x Vector to substract
     * @return Current vector
    **/
    val_t operator*(Vector<dim> const& x) const {
        val_t sum = 0;
        for (nat_t i = 0; i < dim; i++)
            sum += get(i) * x.get(i);
        return sum;
    }
    /** Vector comparison.
     * @param x Vector to compare
     * @return True if equal, false otherwise
    **/
    bool operator==(Vector<dim> const& x) const {
        for (nat_t i = 0; i < dim; i++)
            if (x.get(i) != get(i))
                return false;
        return true;
    }
public:
    /** Compute the p-norm, or the infinite norm.
     * @param p Norm parameter (none for the infinite norm)
     * @return Norm value
    **/
    val_t norm(nat_t p) const {
        val_t pv = static_cast<val_t>(p);
        val_t sum = 0;
        for (nat_t i = 0; i < dim; i++)
            sum += ::std::pow(::std::abs(get(i)), pv);
        return ::std::pow(sum, 1 / pv);
    }
    val_t norm() const {
        val_t max = 0;
        for (nat_t i = 0; i < dim; i++)
            max = ::std::fmax(max, ::std::abs(get(i)));
        return max;
    }
    /** Compute the softmax vector.
     * @return Softmax vector
    **/
    Vector<dim> softmax() {
        Vector<dim> out;
        val_t sum = 0;
        for (nat_t i = 0; i < dim; i++) {
            val_t x = ::std::exp(get(i));
            sum += x;
            out.set(i, x);
        }
        for (nat_t i = 0; i < dim; i++)
            out.set(i, out.get(i) / sum);
        return out;
    }
public:
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    static constexpr size_t size() {
        return dim * sizeof(val_t);
    }
    /** Load vector data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        for (nat_t i = 0; i < dim; i++)
            set(i, input.load());
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) const {
        for (nat_t i = 0; i < dim; i++)
            output.store(get(i));
    }
public:
    /** Print vector to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) const {
        ostr << "{ " << get(0);
        for (nat_t i = 1; i < dim; i++)
            ostr << ", " << get(i);
        ostr << " }";
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Simple vector ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
