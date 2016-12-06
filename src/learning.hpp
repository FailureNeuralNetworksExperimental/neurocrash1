/**
 * @file learning.hpp
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
#include "vector.hpp"
#include "network.hpp"

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Learning discipline ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Learning discipline.
 * @param input_dim  Input vector dimensions
 * @param output_dim Output vector dimensions
**/
template<nat_t input_dim, nat_t output_dim> class Learning final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    /** Input vector type.
    **/
    using Input  = Vector<input_dim>;
    /** Output vector type.
    **/
    using Output = Vector<output_dim>;
    /** Input, and expected output, with error margin.
    **/
    class Constraint final {
    private:
        Input  input;    // Input vector
        Output expected; // Expected output vector
        Output margin;   // Tolerated margin
    public:
        /** Build a new constraint.
         * @param input    Input vector
         * @param expected Expected output vector
         * @param margin   Tolerated margin vector
        **/
        Constraint(Input& input, Output& expected, Output& margin): input(input), expected(expected), margin(margin) {}
    public:
        /** Check equality between input vectors.
         * @param input Input vector to compare with
         * @return True if the stored input vector is equal to the given one, false otherwise
        **/
        bool match(Input& input) {
            return this->input == input;
        }
        /** Correct the network one time, if needed.
         * @param network Neural network to correct
         * @param eta     Correction factor
         * @return True if on bounds, false if a correction has been applied
        **/
        template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> bool correct(Network<Neuron, NeuronLast, implicit_dims...>& network, val_t eta) {
            Output output; // Output vector
            network.compute(input, output);
            for (nat_t i = 0; i < output_dim; i++) { // Check for bounds
                val_t diff = expected.get(i) - output.get(i);
                if ((diff < 0 ? -diff : diff) > margin.get(i)) { // Out of at least one bound
                    network.accumulate(input, expected, output, eta);
                    return false;
                }
            }
            return true;
        }
    public:
        /** Print constraint to the given stream.
         * @param ostr Output stream
        **/
        void print(::std::ostream& ostr) {
            ostr << "{ ";
            input.print(ostr);
            ostr << ", ";
            expected.print(ostr);
            ostr << ", ";
            margin.print(ostr);
            ostr << " }";
        }
    };
private:
    nat_t bs; // Batch size for update
    ::std::vector<Constraint> constraints; // Constraints set
    ::std::random_device device; // Random device
    ::std::default_random_engine engine; // Default engine
public:
    /** Build an empty learning discipline.
     * @param bs Initial batch size
    **/
    Learning(nat_t bs = 1): bs(bs), constraints(), device(), engine(device()) {}
public:
    /** Add a constraint to the discipline, not checked for duplicate.
     * @param input  Input vector
     * @param output Expected output vector
     * @param margin Tolerated margin vector
    **/
    void add(Input& input, Output& output, Output& margin) {
        constraints.emplace(constraints.end(), input, output, margin);
    }
    /** Tell if a constraint exists based on the input vector.
     * @param input Input vector of the constraint to find
     * @return True if a matching constraint has been found, false otherwise
    **/
    bool has(Input& input) {
        for (Constraint& constraint: constraints)
            if (constraint.match(input))
                return true;
        return false;
    }
    /** Remove a constraint based on the input vector.
     * @param input Input vector of the constraint to remove
    **/
    void remove(Input& input) {
        nat_t pos = 0;
        for (Constraint& constraint: constraints) {
            if (constraint.match(input)) {
                constraints.erase(pos);
                return;
            }
            pos++;
        }
    }
    /** Remove all constraints.
    **/
    void reset() {
        constraints.clear();
    }
    /** Get the number of constraints.
     * @return Number of contraints
    **/
    nat_t size() {
        return constraints.size();
    }
    /** Get/modify the batch size.
     * @param nbs New batch size (nothing/0 to not modify)
     * @return Old batch size
    **/
    nat_t batch_size(nat_t nbs = 0) {
        nat_t old = bs;
        if (nbs != 0)
            bs = nbs;
        return old;
    }
public:
    /** Correct the network one time, so that each output is near enough from its expected output.
     * @param network Neural network to correct
     * @param eta     Correction factor
     * @return Number of out-bounds constraints
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> nat_t correct(Network<Neuron, NeuronLast, implicit_dims...>& network, val_t eta) {
        nat_t count = 0;
        for (Constraint& constraint: constraints) {
            if (!constraint.correct(network, eta)) { // Not in-bounds
                count++;
                if (count % bs == 0) // Number of correction reached
                    network.update(bs); // Apply accumulated corrections
            }
        }
        if (count % bs != 0) // Something to update
            network.update(count % bs); // Apply remaining corrections
        return count;
    }
    /** Randomize constraints order.
    **/
    void shuffle() {
        ::std::shuffle(constraints.begin(), constraints.end(), engine);
    }
public:
    /** Print learning discipline to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
        if (constraints.empty()) {
            ostr << "{}";
            return;
        }
        ostr << "{" << ::std::endl << "\t";
        bool first = true;
        for (Constraint& constraint: constraints) {
            if (first) {
                first = false;
                constraint.print(ostr);
            } else {
                ostr << "," << ::std::endl << "\t";
                constraint.print(ostr);
            }
        }
        ostr << ::std::endl << "}";
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Learning discipline ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
