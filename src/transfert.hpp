/**
 * @file transfert.hpp
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
// ▁ Print transfert functions ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Print a transfert function.
 * @param Transfert Transfert function to print
**/
template<class Transfert> class TransfertPrinter final {
public:
    /** Print functions (transfert, transfert derivative) to the given stream, to plot them.
     * @param ostr  Output stream
     * @param from  Start abscissa
     * @param to    End abscissa (undefined behavior if to <= from)
     * @param count Amount of points
    **/
    static void print(::std::ostream& ostr, val_t from, val_t to, nat_t count) {
        val_t const delta = to - from;
        val_t const prec = static_cast<val_t>(count);
        for (nat_t i = 0; i <= count; i++) { // Base and diff
            val_t x = delta * static_cast<val_t>(i) / prec + from;
            ostr << x << "\t" << Transfert::calc(x) << "\t" << Transfert::diff(x) << ::std::endl;
        }
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Print transfert functions ▔
// ▁ Precomputed transfert function ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Precomputed transfert function.
**/
class PrecomputedTransfert final {
private:
    static constexpr val_t diff_delta = 0.01; // Delta for derivative estimation
private:
    nat_t  count; // Nb points - 1
    val_t  x_min; // Min input
    val_t  x_max; // Max input
    val_t  delta; // Difference between max/min
    val_t* tbase; // Points of base function (null if not initialized)
    val_t* tdiff; // Points of derived function (= base + prec)
private:
    /** Function type selector.
    **/
    enum class select { base, diff }; // Function type selector
    /** Select a table.
     * @param func Table to select
     * @return Selected table
    **/
    template<select func> val_t* get() const {
        switch (func) {
            case select::base:
                return tbase;
            case select::diff:
                return tdiff;
        }
    }
    /** Interpolate (linearly) a value through this function/its derivative/other.
     * @param func Selected function ('base' or 'diff')
     * @param x    Input value
     * @return Output value
    **/
    template<select func> val_t interpolate(val_t x) const {
        if (unlikely(x < x_min)) {
            return get<func>()[0];
        } else if (unlikely(x >= x_max)) {
            return get<func>()[count];
        } // Else linear interpolation
        val_t t = (x - x_min) * static_cast<val_t>(count) / delta;
        nat_t i = static_cast<nat_t>(t); // Truncated
        if (unlikely(i >= count)) { // Due to floating-point imprecision
            return get<func>()[count];
        } else {
            val_t y_a = get<func>()[i];
            val_t y_b = get<func>()[i + 1];
            return y_a + (y_b - y_a) * ::std::fmod(t, val_t(1));
        }
    }
public:
    /** Constructor.
    **/
    PrecomputedTransfert(): tbase(null) {}
    /** Destructor.
    **/
    ~PrecomputedTransfert() {
        if (tbase)
            ::std::free(static_cast<void*>(tbase));
    }
public:
    /** Pass parameter through the transfert function.
     * @param x Input value
     * @return Output value
    **/
    val_t operator()(val_t x) const {
        return interpolate<select::base>(x);
    }
    /** Pass parameter through the transfert function derivative.
     * @param x Input value
     * @return Output value
    **/
    val_t diff(val_t x) const {
        return interpolate<select::diff>(x);
    }
    /** (Re)set the transfert function, with optional weight correction.
     * @param trans Transfert function
     * @param min   Min input
     * @param max   Max input
     * @param prec  Amount of points
     * @param corr  Weight correction (optional)
     * @return True if the operation is a success, false otherwise
    **/
    bool set(::std::function<val_t(val_t)> const& trans, val_t min, val_t max, nat_t prec) {
        if (unlikely(min >= max || prec < 2)) // Basic checks
            return false;
        { // Points table allocation
            if (tbase) // Points table freeing (if already exists)
                ::std::free(static_cast<void*>(tbase));
            void* addr = ::std::malloc(2 * prec * sizeof(val_t));
            if (!addr) { // Allocation failure
                tbase = null;
                return false;
            }
            tbase = static_cast<val_t*>(addr);
            tdiff = tbase + prec;
        }
        { // Tables initialization
            x_min = min;
            x_max = max;
            delta = max - min;
            count = prec - 1;
            val_t const prec1 = static_cast<val_t>(prec - 1);
            for (nat_t i = 0; i < prec; i++) { // Base and diff
                val_t x = delta * static_cast<val_t>(i) / prec1 + x_min;
                tbase[i] = trans(x);
                tdiff[i] = (trans(x + diff_delta / 2) - trans(x - diff_delta / 2)) / diff_delta;
            }
        }
        return true;
    }
    bool set(::std::function<val_t(val_t)>&& trans, val_t min, val_t max, nat_t prec) { // So that we can "emplace" the lambda directly in the function call
        return set(trans, min, max, prec);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Precomputed transfert function ▔
// ▁ Basic transfert functions ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Linear transfert function.
**/
class LinearTransfert {
public:
    static val_t k; // Input factor
public:
    /** Compute f(x).
     * @param x Parameter
     * @return k * x
    **/
    static val_t calc(val_t x) {
        return k * x;
    }
    /** Compute f'(x).
     * @param x Parameter
     * @return 1
    **/
    static val_t diff(val_t x) {
        return k;
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Rectifier transfert function.
**/
class RectifierTransfert {
public:
    static val_t k; // Input factor
public:
    /** Compute f(x).
     * @param x Parameter
     * @return max(0, k * x)
    **/
    static val_t calc(val_t x) {
        return (x > 0 ? (k * x) : 0);
    }
    /** Compute f'(x).
     * @param x Parameter
     * @return d(max(0, k * x))/dx, 0 if x == 0
    **/
    static val_t diff(val_t x) {
        return (x > 0 ? k : 0);
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Sigmoid transfert function.
**/
class SigmoidTransfert {
public:
    static val_t k; // Input factor
public:
    /** Compute f(x).
     * @param x Parameter
     * @return f(x)
    **/
    static val_t calc(val_t x) {
        return val_t(1) / (val_t(1) + ::std::exp(-k * x));
    }
    /** Compute f'(x).
     * @param x Parameter
     * @return f'(x)
    **/
    static val_t diff(val_t x) {
        val_t e = ::std::exp(-k * x);
        val_t d = 1 + e;
        return (k * e) / (d * d);
    }
};

/** Precomputed sigmoid transfert function.
**/
class PrecomputedSigmoidTransfert {
private:
    static PrecomputedTransfert sigmoid; // Transfert function
public:
    /** Initialize the transfert function.
     * @param k     Input factor (= slope)
     * @param from  Start abscissa
     * @param to    End abscissa (undefined behavior if to <= from)
     * @param count Amount of points
     * @return True on success, false otherwise
    **/
    static bool init(val_t k, val_t from = -5, val_t to = 5, nat_t count = 1001) {
        auto const transfert_function = [k](val_t x) -> val_t { return val_t(1) / (val_t(1) + ::std::exp(-k*x)); }; // Sigmoid
        if (!sigmoid.set(transfert_function, from, to, count)) {
            ::std::cerr << "Precache of the transfert function failed" << ::std::endl;
            return false;
        }
        return true;
    }
public:
    /** Compute f(x).
     * @param x Parameter
     * @return f(x)
    **/
    static val_t calc(val_t x) {
        return sigmoid(x);
    }
    /** Compute f'(x).
     * @param x Parameter
     * @return f'(x)
    **/
    static val_t diff(val_t x) {
        return sigmoid.diff(x);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Basic transfert functions ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
