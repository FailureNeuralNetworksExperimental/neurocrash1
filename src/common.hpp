/**
 * @file common.hpp
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

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

namespace StaticNet {

// Types used
using val_t = float;         // Any floating-point value
using nat_t = uint_fast32_t; // Any natural number

// Null value (I prefer null over nullptr)
auto const null = nullptr;

/** Specify a proposition as 'likely true'.
 * @param prop Proposition likely true
**/
#undef likely
#ifdef __GNUC__
    #define likely(prop) \
        __builtin_expect((prop) ? true : false, true)
#else
    #define likely(prop) \
        (prop)
#endif

/** Specify a proposition as 'likely false'.
 * @param prop Proposition likely false
**/
#undef unlikely
#ifdef __GNUC__
    #define unlikely(prop) \
        __builtin_expect((prop) ? true : false, false)
#else
    #define unlikely(prop) \
        (prop)
#endif

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Random number generator ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Abstract randomizer.
**/
class Randomizer {
public:
    /** Get a random number.
    **/
    virtual val_t get() = 0;
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Uniform distribution randomizer.
**/
class UniformRandomizer final: public Randomizer {
private:
    ::std::random_device                    device;
    ::std::default_random_engine            engine;
    ::std::uniform_real_distribution<val_t> distrib;
public:
    /** Constructor.
     * @param limit Uniform distribution over [-limit, +limit], must be positive
    **/
    UniformRandomizer(val_t limit): device(), engine(device()), distrib(-limit, limit) {}
public:
    /** Get a random number.
     * @return A random number
    **/
    val_t get() {
        return distrib(engine);
    }
};

/** Gaussian distribution randomizer.
**/
class GaussianRandomizer final: public Randomizer {
private:
    ::std::random_device              device;
    ::std::default_random_engine      engine;
    ::std::normal_distribution<val_t> distrib;
public:
    /** Constructor.
     * @param mean   Distribution mean
     * @param stddev Distribution standard deviation
    **/
    GaussianRandomizer(val_t mean, val_t stddev): device(), engine(device()), distrib(mean, stddev) {}
public:
    /** Get a random number.
     * @return A random number
    **/
    val_t get() {
        return distrib(engine);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Random number generator ▔
// ▁ Input/Output serializer ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {
namespace Serializer {

/** Abstract input serializer class.
**/
class Input {
public:
    /** Load one value, in order of writing.
     * @return Value loaded
    **/
    virtual val_t load() = 0;
};

/** Abstract output serializer class.
**/
class Output {
public:
    /** Store one value.
     * @param Value stored
    **/
    virtual void store(val_t) = 0;
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Input serializer based on a stream.
**/
class StreamInput: public Input {
private:
    ::std::istream& istream; // Input stream
public:
    /** Build a simple input stream.
     * @param istream Input stream to use
    **/
    StreamInput(::std::istream& istream): istream(istream) {}
public:
    /** Load one value.
     * @return value Value stored
    **/
    virtual val_t load() {
        val_t value;
        istream.read(reinterpret_cast<::std::remove_reference<decltype(istream)>::type::char_type*>(&value), sizeof(val_t));
        return value;
    }
};

/** Output serializer based on a stream.
**/
class StreamOutput: public Output {
private:
    ::std::ostream& ostream; // Output stream
public:
    /** Build a simple output stream.
     * @param ostream Output stream to use
    **/
    StreamOutput(::std::ostream& ostream): ostream(ostream) {}
public:
    /** Store one value.
     * @param value Value stored
    **/
    virtual void store(val_t value) {
        ostream.write(reinterpret_cast<::std::remove_reference<decltype(ostream)>::type::char_type*>(&value), sizeof(val_t));
    }
};

} }

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Input/Output serializer ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
