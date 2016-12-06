/**
 * @file rand.cpp
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

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
extern "C" {
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
}

// Internal headers
#include <staticnet.hpp>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

using namespace StaticNet;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Constants ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// Preprocessors
#define dimensions input_dim, first_layer, second_layer, third_layer, fourth_layer, output_dim // Network dimensions
#define killlayer 1 // In which layer to kill neurons
#if killlayer == 1
    #define robust_kf_val 0
    #define robust_kt_val first_layer
#elif killlayer == 2
    #define robust_kf_val first_layer
    #define robust_kt_val second_layer
#elif killlayer == 3
    #define robust_kf_val (first_layer + second_layer)
    #define robust_kt_val third_layer
#else
    #error Invalid killlayer value
#endif

// Constants
static constexpr nat_t input_dim    = 4; // Input space dimension
static constexpr nat_t output_dim   = 1; // Output space dimension
static constexpr nat_t first_layer  = 4; // First layer output dimension (= #neurons)
static constexpr nat_t second_layer = 4; // Second layer output dimension (= #neurons)
static constexpr nat_t third_layer  = 4; // Second layer output dimension (= #neurons)
static constexpr nat_t fourth_layer = 4; // Second layer output dimension (= #neurons)
static constexpr nat_t robust_killfrom = robust_kf_val; // Offset in the killmap
static constexpr nat_t robust_killto   = robust_kt_val; // Length kill from the offset in the killmap

/** Input vector.
**/
using Input = Vector<input_dim>;

/** Output vector.
**/
using Output = Vector<output_dim>;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Constants ▔
// ▁ Networks ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Hidden transfert function.
**/
using HiddenTransfert = RectifierTransfert;

/** Hidden transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using HiddenNeuron = StandardNeuron<HiddenTransfert, false>::Neuron<input_dim>;

/** Output transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using OutputNeuron = StandardNeuron<LinearTransfert, false>::Neuron<input_dim>;

/** Default network used.
**/
using NetworkDefault = Network<HiddenNeuron, OutputNeuron, dimensions>;

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Transferts
val_t LinearTransfert::k = 1;
val_t HiddenTransfert::k;

/** Init transfert functions.
 * @param k Input factor
 * @return True on success, false otherwise
**/
static bool transfert_init(val_t k) {
    HiddenTransfert::k = k;
    return true;
}

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Map of killed neurons
static bool killmap[NetworkDefault::length()];

/** Killable neuron.
 * @param LowerNeuron Lower neuron used
**/
template<template<nat_t> class LowerNeuron> class KillableNeuron final {
public:
    /** Killable neuron.
     * @param input_dim Input vector dimension
    **/
    template<nat_t input_dim> class Neuron: public LowerNeuron<input_dim> {
    private:
        nat_t id; // Neuron ID
    public:
        /** Set the ID of the neuron.
         * @param id ID of the neuron
        **/
        void as(nat_t new_id) {
            id = new_id;
        }
        /** Compute the output of the neuron, return 0 if it has been killed.
         * @param input Input vector
         * @return Output scalar
        **/
        val_t compute(Vector<input_dim> const& input) {
            return killmap[id] ? 0 : LowerNeuron<input_dim>::compute(input);
        }
    };
};

/** Killable neuron network.
**/
using NetworkKillable = Network<KillableNeuron<HiddenNeuron>::Neuron, OutputNeuron, dimensions>;

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// {Maximum, average, std dev} absolute value transported by a synapse, per layer
static val_t synapse_max[NetworkDefault::layer_length()] = { 0 };
static val_t synapse_avg[NetworkDefault::layer_length()] = { 0 };
static val_t synapse_std[NetworkDefault::layer_length()] = { 0 };

// {Maximum, average, std dev} absolute weight per layer (does not count bias)
static val_t weight_max[NetworkDefault::layer_length()] = { 0 };
static val_t weight_avg[NetworkDefault::layer_length()] = { 0 };
static val_t weight_std[NetworkDefault::layer_length()] = { 0 };

/** Layer info neuron.
 * @param LowerNeuron Lower neuron used
**/
template<template<nat_t> class LowerNeuron> class InfoNeuron final {
public:
    /** Layer info neuron.
     * @param input_dim Input vector dimension
    **/
    template<nat_t input_dim> class Neuron: public LowerNeuron<input_dim> {
    private:
        nat_t id; // Layer ID
    public:
        /** Set the ID of the neuron.
         * @param id ID of the neuron
        **/
        void as(nat_t new_id) {
            id = NetworkDefault::neuron_to_layer(new_id);
        }
        /** Compute the output of the neuron, return 0 if it has been killed.
         * @param input Input vector
         * @return Output scalar
        **/
        val_t compute(Vector<input_dim> const& input) {
            for (nat_t i = 0; i < input_dim; i++) {
                val_t value = input.get(i);
                if (value < 0)
                    value = -value;
                if (value > synapse_max[id]) // Larger synaptic value for the current layer
                    synapse_max[id] = value;
                synapse_avg[id] += value / static_cast<val_t>(NetworkDefault::weight_length());
                synapse_std[id] += (value * value) / static_cast<val_t>(NetworkDefault::weight_length());
            }
            return LowerNeuron<input_dim>::compute(input);
        }
    public:
        /** Load neuron data.
         * @param input Serialized input
        **/
        void load(Serializer::Input& input) {
            LowerNeuron<input_dim>::load(input);
            for (nat_t i = 0; i < input_dim; i++) {
                val_t w = LowerNeuron<input_dim>::weight.get(i);
                if (w < 0)
                    w = -w;
                if (w > weight_max[id]) // Larger weight value for the current layer
                    weight_max[id] = w;
                weight_avg[id] += w / static_cast<val_t>(NetworkDefault::weight_length());
                weight_std[id] += (w * w) / static_cast<val_t>(NetworkDefault::weight_length());
            }
        }
    };
};

/** Network measuring layer-related information.
**/
using NetworkLayerInfo = Network<InfoNeuron<HiddenNeuron>::Neuron, InfoNeuron<OutputNeuron>::Neuron, dimensions>;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Networks ▔
// ▁ Simple transformations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace Helper {

/** Compute the number of combinations.
 * @param n Total number of elements
 * @param k Selected number of elements, k <= n
 * @return C^k_n
**/
static nat_t combination(nat_t n, nat_t k) {
    nat_t d = n - k;
    if (d < k) { // We want d to be bigger that k
        nat_t t = d;
        d = k;
        k = t;
    }
    nat_t r = 1;
    for (nat_t i = d + 1; i <= n; i++)
        r *= i;
    for (nat_t i = 2; i <= k; i++)
        r /= i;
    return r;
}

/** Compute the power of an integer.
 * @param n Integer
 * @param p Power
 * @return n^p
**/
static constexpr nat_t power(nat_t n, nat_t p) {
    nat_t out = 1;
    for (nat_t i = 0; i < p; i++)
        out *= n;
    return out;
}

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Buffered input stream, that can replay a stream.
**/
class BufferedStreamInput final: public Serializer::StreamInput {
private:
    bool  record; // True if the stream is being recorded, false otherwise
    nat_t pos;    // Position in the stream
    ::std::vector<val_t> buffer; // Buffer of the stream
public:
    /** Build a buffered input stream.
     * @param istream Input stream to use
    **/
    BufferedStreamInput(::std::istream& istream): Serializer::StreamInput(istream), record(true) {}
public:
    /** Load one value.
     * @return value Value stored
    **/
    val_t load() {
        if (record) {
            val_t value = Serializer::StreamInput::load();
            buffer.push_back(value);
            return value;
        } else {
            return buffer[pos++];
        }
    }
    /** Rewind the stream to be replayed.
    **/
    void rewind() {
        record = false;
        pos = 0;
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Simple dimensions printer.
 * @param ... Dimensions
**/
template<nat_t dim, nat_t... dims> class DimensionsPrinter final {
public:
    /** Build the network dimensions string.
     * @param separator Separator to use
     * @return Dimensions string
    **/
    static ::std::string to_string(::std::string const& separator) {
        return ::std::to_string(dim) + separator + DimensionsPrinter<dims...>::to_string(separator);
    }
};
template<nat_t dim> class DimensionsPrinter<dim> final {
public:
    /** Build the network dimensions string.
     * @param separator Separator to use
     * @return Dimensions string
    **/
    static ::std::string to_string(::std::string const& separator) {
        return ::std::to_string(dim);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Simple transformations ▔
// ▁ Measurements ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Measurements.
**/
class Measure final {
private:
    /** Input generator.
    **/
    class InputGenerator final {
    public:
        static constexpr nat_t digit = 101; // Number of divisions per coordinate
        static constexpr val_t limit = 1;   // Coordinate segment limit (each coord. in [0, limit])
        static constexpr nat_t length = Helper::power(digit, input_dim); // Limit vector
    public:
        /** Set a vector with a given entry.
         * @param vec Vector to set
         * @param id  Vector ID
        **/
        static void set(Input& vec, nat_t id) {
            for (nat_t i = 0; i < input_dim; i++) {
                vec.set(i, static_cast<val_t>(id % digit) / static_cast<val_t>(digit - 1) * limit);
                id /= digit;
            }
        }
    };
    /** Return block.
    **/
    class Omega final {
    public:
        val_t max; // Max value
        val_t avg; // Average value
        val_t std; // Element of standard deviation computation
    public:
        /** Zero-constructor.
        **/
        Omega(): max(0), avg(0), std(0) {}
    };
public:
    /** Compute the synaptic info per layer by running the network on many different inputs.
     * @param network Network to use
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> static void synapse_limits(Network<Neuron, NeuronLast, implicit_dims...>& network) {
        Input input;
        for (nat_t i = 0; i < InputGenerator::length; i++) { // For each possible input
            InputGenerator::set(input, i);
            Output result;
            network.compute(input, result);
        }
        for (nat_t i = 0; i < NetworkDefault::layer_length(); i++) { // Finalization of averages and standard deviations
            synapse_avg[i] /= static_cast<val_t>(InputGenerator::length);
            synapse_std[i] /= static_cast<val_t>(InputGenerator::length);
            synapse_std[i] = ::std::sqrt(synapse_std[i] - synapse_avg[i] * synapse_avg[i]);
        }
    }
    /** Compute the omega value of the network.
     * @param network   Network to use
     * @param reference Network reference to use
     * @return Value of max omega, value of average omega on the whole test set, value of average square omega on the whole test set
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, template<nat_t> class NeuronRef, template<nat_t> class NeuronRefLast, nat_t... implicit_dims> static ::std::tuple<Omega, Omega> omega(Network<Neuron, NeuronLast, implicit_dims...>& network, Network<NeuronRef, NeuronRefLast, implicit_dims...>& reference) {
        Omega simple;
        Omega normalized;
        val_t card = InputGenerator::length;
        Input input;
        for (nat_t i = 0; i < InputGenerator::length; i++) { // For each possible input
            InputGenerator::set(input, i);
            Output result;
            Output expected;
            network.compute(input, result);
            reference.compute(input, expected);
            result -= expected; // Simple delta
            { // Simple omega
                val_t norm = result.norm(); // Infinite norm of difference
                if (norm > simple.max)
                    simple.max = norm;
                simple.avg += norm / card;
                simple.std += (norm * norm) / card;
            }
            { // Normalized omega
                val_t div = expected.norm();
                if (div > 0) {
                    val_t norm = result.norm() / div; // Infinite norm of normalized difference
                    if (norm > normalized.max)
                        normalized.max = norm;
                    normalized.avg += norm / card;
                    normalized.std += (norm * norm) / card;
                }
            }
        }
        return ::std::make_tuple(simple, normalized);
    }
    /** Compute robustness-related metrics (epsilon and omega).
     * @param network   Network to use
     * @param reference Network reference for omega computation
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, template<nat_t> class NeuronRef, template<nat_t> class NeuronRefLast, nat_t... implicit_dims> static void robust(Network<Neuron, NeuronLast, implicit_dims...>& network, Network<NeuronRef, NeuronRefLast, implicit_dims...>& reference) {
        nat_t const offset_limit = robust_killto; // Amount of killable neurons
        for (nat_t i = 0; i < NetworkDefault::length(); i++) // Reset the entire map
            killmap[i] = false;
        bool* killview = killmap + robust_killfrom; // Set of neurons potentially killed
        for (nat_t i = 0; i <= offset_limit; i++) {
            Omega global_simple;
            Omega global_normalized; /// FIXME: Méthode de calcul instable
            for (nat_t j = 0; j < offset_limit; j++) // Initialize the array
                killview[j] = (j < i ? true : false);
            nat_t combs = Helper::combination(offset_limit, i);
            for (nat_t j = 0;;) { // For all possible combinations
                if (i != 0 || j != 0) { // "Partially compute" omega_max and omega_avg (avoid case where no neuron is killed)
                    Omega local_simple;
                    Omega local_normalized;
                    ::std::tie(local_simple, local_normalized) = omega(network, reference);
                    { // Simple
                        if (local_simple.max > global_simple.max)
                            global_simple.max = local_simple.max;
                        global_simple.avg += local_simple.avg / static_cast<val_t>(combs);
                        global_simple.std += local_simple.std / static_cast<val_t>(combs);
                    }
                    { // Normalized
                        if (local_normalized.max > global_normalized.max)
                            global_normalized.max = local_normalized.max;
                        global_normalized.avg += local_normalized.avg / static_cast<val_t>(combs);
                        global_normalized.std += local_normalized.std / static_cast<val_t>(combs);
                    }
                }
                if (++j >= combs) // All combinations have been seen ?
                    break;
                nat_t passed = 0; // Number of elements "passed"
                bool passing = true; // Passing room check
                for (nat_t k = offset_limit - 1;; k--) { // Get to next position
                    if (passing) {
                        if (killview[k]) { // Not more room
                            killview[k] = false;
                            passed++;
                        } else {
                            passing = false;
                        }
                    } else if (killview[k]) { // Element to move
                        killview[k] = false;
                        killview[k + 1] = true;
                        for (nat_t l = k + 2; passed > 0; l++) {
                            killview[l] = true;
                            passed--;
                        }
                        break; // End of operations
                    }
                }
            }
            global_simple.std = ::std::sqrt(global_simple.std - global_simple.avg * global_simple.avg);
            global_normalized.std = ::std::sqrt(global_normalized.std - global_normalized.avg * global_normalized.avg);
            ::std::cout << i << "\t"
                << global_simple.max << "\t" << global_simple.avg << "\t" << global_simple.std << "\t"
                << global_normalized.max << "\t" << global_normalized.avg << "\t" << global_normalized.std << ::std::endl;
        }
    }
};

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Measurements ▔
// ▁ Orders ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Random network generation.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int gen_rand(int argc, char** argv) {
    if (argc != 2 && argc != 4) { // Wrong number of parameters
        ::std::cerr << "Usage: " << argv[0] << " " << argv[1]  << " [<mean> <stddev>] | 'raw random network'" << ::std::endl;
        return 0;
    }
    NetworkDefault network;
    { // Generation phase
        val_t mean;   // Gaussian distribution mean
        val_t stddev; // Gaussian distribution standard deviation
        if (argc == 4) { // Custom values
            mean = ::std::atof(argv[2]);
            stddev = ::std::atof(argv[3]);
        } else { // Default values
            mean = 1;
            stddev = 5;
        }
        GaussianRandomizer gauss(mean, stddev); // Gaussian distribution
        network.randomize(gauss);
    }
    { // Output phase
        Serializer::StreamOutput so(::std::cout);
        network.store(so);
    }
    return 0;
}

/** Robustness-related measurements.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int robust(int argc, char** argv) {
    if (argc != 3) { // Wrong number of parameters
        ::std::cerr << "Usage: 'raw random network' | " << argv[0] << " " << argv[1]  << " <k-param> | 'robustness data'" << ::std::endl;
        return 0;
    }
    if (!transfert_init(::std::atof(argv[2]))) // Initialize transfert functions
        return 1;
    NetworkDefault  reference; // Using default network as reference for computation of omega
    NetworkKillable network;   // Using network with killable neurons
    { // Input phase
        Helper::BufferedStreamInput bsi(::std::cin);
        reference.load(bsi);
        bsi.rewind();
        network.load(bsi);
    }
    { // Measurement phase
        ::std::cerr << "Robustness measurements...";
        ::std::cerr.flush();
        Measure::robust(network, reference);
        ::std::cerr << " done." << ::std::endl;
    }
    return 0;
}

/** Compute, per layer, the {maximum, average, std dev} {absolute weight, absolute value transported by a synapse}.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int layer_info(int argc, char** argv) {
    if (argc != 3) { // Wrong number of parameters
        ::std::cerr << "Usage: 'raw random network' | " << argv[0] << " " << argv[1]  << " <k-param> | 'layer info data'" << ::std::endl;
        return 0;
    }
    if (!transfert_init(::std::atof(argv[2]))) // Initialize transfert functions
        return 1;
    NetworkLayerInfo network; // Using network givin layer info
    { // Input phase
        Serializer::StreamInput si(::std::cin);
        network.load(si); // Computes weigth_*
        for (nat_t i = 0; i < NetworkDefault::layer_length(); i++) // Finalization of averages and standard deviations
            weight_std[i] = ::std::sqrt(weight_std[i] - weight_avg[i] * weight_avg[i]);
    }
    { // Measurement phase
        ::std::cerr << "Layer info measurements...";
        ::std::cerr.flush();
        Measure::synapse_limits(network);
        ::std::cerr << " done." << ::std::endl;
    }
    { // Output phase
        for (nat_t i = 0; i < NetworkDefault::layer_length(); i++)
            ::std::cout << i << "\t" << synapse_max[i] << "\t" << synapse_avg[i] << "\t" << synapse_std[i] << "\t" << weight_max[i] << "\t" << weight_avg[i] << "\t" << weight_std[i] << ::std::endl;
    }
    return 0;
}

/** Print transfert functions, in a way easily plot-able by gnuplot.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int plot(int argc, char** argv) {
    if (argc != 3 && argc != 5 && argc != 6) { // Wrong number of parameters
        ::std::cerr << "Usage: " << argv[0] << " " << argv[1]  << " <k-param> [from to [count]]  | 'plot points'" << ::std::endl;
        return 0;
    }
    if (!transfert_init(::std::atof(argv[2]))) // Initialize transfert functions
        return 1;
    val_t from  = -6;
    val_t to    = +6;
    nat_t count = 1001;
    switch (argc) {
    case 6:
        count = ::std::strtoul(argv[5], null, 10);
    case 5:
        from = ::std::atof(argv[3]);
        to = ::std::atof(argv[4]);
    }
    TransfertPrinter<HiddenTransfert>::print(::std::cout, from, to, count);
    return 0;
}

/** Simply output supported network dimensions.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int print_dims(int argc, char** argv) {
    if (argc < 2 || argc > 3) { // Wrong number of parameters
        ::std::cerr << "Usage: " << argv[0] << " " << argv[1] << " [separator] | 'supported network dimensions'" << ::std::endl;
        return 0;
    }
    ::std::cout << Helper::DimensionsPrinter<dimensions>::to_string(argc == 3 ? argv[2] : "-") << ::std::endl;
    return 0;
}

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Map order to handler
static ::std::unordered_map<::std::string, int (*)(int, char**)> orders = {
    { "rand", gen_rand },
    { "robust", robust },
    { "layer_info", layer_info },
    { "plot", plot },
    { "dimensions", print_dims }
};

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Orders ▔
// ▁ Entry point ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Program entry point.
 * @param argc Number of arguments
 * @param argv Arguments
 * @return Return code
**/
int main(int argc, char** argv) {
    if (argc < 2 || orders.count(argv[1]) == 0) { // Wrong number of parameters or unknown order
        ::std::cerr << "Usage: " << (argc != 0 ? argv[0] : "mnist") << " {";
        bool first = true;
        for (auto const& i: orders) {
            if (first) {
                first = false;
            } else {
                ::std::cerr << " | ";
            }
            ::std::cerr << i.first;
        }
        ::std::cerr << "}" << ::std::endl;
        return 0;
    }
    return orders[argv[1]](argc, argv); // Order handling
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Entry point ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
