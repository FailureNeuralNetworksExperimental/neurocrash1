/**
 * @file network.hpp
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
#include "transfert.hpp"

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Neurons ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Single neuron with synapses.
 * @param Transfert Transfert function class
 * @param use_bias  True to use a bias (optional, op-in)
**/
template<class Transfert, bool use_bias = true> class StandardNeuron {
public:
    /** Single neuron with synapses.
     * @param input_dim Input vector dimension
    **/
    template<nat_t input_dim> class Neuron {
        static_assert(input_dim > 0, "Invalid input vector dimension");
    protected:
        Vector<input_dim> weight; // Input weight vector
        val_t bias; // Bias
        val_t sum;  // Last weighted sum of inputs
        Vector<input_dim> delta_weight; // Input weight correction sum
        val_t delta_bias; // Bias correction sum
    public:
        /** Delta-null constructor.
        **/
        Neuron(): delta_weight(0), delta_bias(0) {}
    public:
        /** Set the ID of the neuron, does nothing.
         * @param id ID of the neuron (ignored)
        **/
        void as(nat_t new_id) {}
        /** Randomize the weight vector.
         * @param rand Randomizer to use
        **/
        void randomize(Randomizer& rand) {
            for (nat_t i = 0; i < input_dim; i++)
                weight.set(i, rand.get());
            if (use_bias)
                bias = rand.get();
        }
        /** Get the weight vector.
         * @return Weight vector
        **/
        Vector<input_dim>& get_weight() {
            return weight;
        }
        /** Compute the output of the neuron.
         * @param input Input vector
         * @return Output scalar
        **/
        val_t compute(Vector<input_dim> const& input) {
            sum = weight * input;
            if (use_bias)
                sum += bias;
            return Transfert::calc(sum);
        }
        /** Accumulate (sum) the corrections on the weight vector of the neuron, must be done right after 'compute'.
         * @param input Input vector
         * @param error Error on output
         * @param eta   Correction factor
         * @return Error scalar for this neuron
        **/
        val_t accumulate(Vector<input_dim> const& input, val_t error, val_t eta) {
            val_t err = error * Transfert::diff(sum);
            for (nat_t i = 0; i < input_dim; i++)
                delta_weight.set(i, delta_weight.get(i) + eta * err * input.get(i));
            if (use_bias)
                delta_bias += eta * err;
            return err;
        }
        /** Apply the accumulated correction on the weight vector of the neuron.
         * @param count Number of accumulated corrections
        **/
        void update(nat_t count) {
            for (nat_t i = 0; i < input_dim; i++)
                weight.set(i, weight.get(i) + delta_weight.get(i) / static_cast<val_t>(count));
            delta_weight = 0;
            if (use_bias) {
                bias += delta_bias / static_cast<val_t>(count);
                delta_bias = 0;
            }
        }
    public:
        /** Return the size of the structure.
         * @return Size of the structure, in bytes
        **/
        static constexpr size_t size() {
            return decltype(weight)::size() + sizeof(val_t);
        }
        /** Load neuron data.
         * @param input Serialized input
        **/
        void load(Serializer::Input& input) {
            weight.load(input);
            if (use_bias)
                bias = input.load();
        }
        /** Store vector data.
         * @param output Serialized output
        **/
        void store(Serializer::Output& output) const {
            weight.store(output);
            if (use_bias)
                output.store(bias);
        }
    public:
        /** Print neuron weights to the given stream.
         * @param ostr Output stream
        **/
        void print(::std::ostream& ostr) const {
            ostr << "{ ";
            weight.print(ostr);
            ostr << ", " << (use_bias ? bias : 0) << " }";
        }
    };
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Linear transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using LinearNeuron = StandardNeuron<LinearTransfert>::Neuron<input_dim>;

/** Rectifier transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using RectifierNeuron = StandardNeuron<RectifierTransfert>::Neuron<input_dim>;

/** Sigmoid transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using SigmoidNeuron = StandardNeuron<SigmoidTransfert>::Neuron<input_dim>;

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Neurons ▔
// ▁ Layer ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Layer of neurons.
 * @param Neuron     Neuron used
 * @param input_dim  Input vector dimension
 * @param output_dim Output vector dimension
**/
template<template<nat_t> class Neuron, nat_t input_dim, nat_t output_dim> class Layer final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    Neuron<input_dim> neurons[output_dim]; // Neurons
public:
    /** Neuron IDs dispatcher constructor.
     * @param offset Offset for the neuron IDs
    **/
    Layer(nat_t offset) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].as(offset + i);
    }
public:
    /** Randomize the layer.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].randomize(rand);
    }
    /** Compute the output vector of the layer.
     * @param input  Input vector
     * @param output Output vector
    **/
    void compute(Vector<input_dim> const& input, Vector<output_dim>& output) {
        for (nat_t i = 0; i < output_dim; i++)
            output.set(i, neurons[i].compute(input));
    }
    /** Correct the neurons of the layer.
     * @param input Input vector
     * @param error Sum of weighted errors vector
     * @param eta   Correction factor
     * @param error_out Sum of weighted errors vector (optional, because useless for the first layer)
    **/
    void accumulate(Vector<input_dim> const& input, Vector<output_dim> const& error, val_t eta, Vector<input_dim>* error_out = null) {
        if (error_out) { // Error vector asked
            Vector<output_dim> errors; // Neuron errors
            for (nat_t i = 0; i < output_dim; i++)
                errors.set(i, neurons[i].accumulate(input, error.get(i), eta));
            for (nat_t i = 0; i < input_dim; i++) { // Compute error vector
                val_t sum = 0; // Sum of weighted error
                for (nat_t j = 0; j < output_dim; j++)
                    sum += neurons[j].get_weight().get(i) * errors.get(j);
                error_out->set(i, sum);
            }
        } else {
            for (nat_t i = 0; i < output_dim; i++)
                neurons[i].accumulate(input, error.get(i), eta);
        }
    }
    /** Apply the accumulated correction on the weight vector of all neurons.
     * @param count Number of accumulated corrections
    **/
    void update(nat_t count) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].update(count);
    }
public:
    /** Return the number of neurons in this layer.
     * @return Number of neurons
    **/
    static constexpr nat_t length() {
        return output_dim;
    }
    /** Return the number of weights in this layer.
     * @return Number of weights (does not count bias)
    **/
    static constexpr nat_t weight_length() {
        return input_dim * output_dim;
    }
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    static constexpr size_t size() {
        size_t size = 0;
        for (nat_t i = 0; i < output_dim; i++)
            size += decltype(*neurons)::size();
        return size;
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) const {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) const {
        ostr << "{" << ::std::endl << "\t";
        neurons[0].print(ostr);
        for (nat_t i = 1; i < output_dim; i++) {
            ostr << "," << ::std::endl << "\t";
            neurons[i].print(ostr);
        }
        ostr << ::std::endl << "}";
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Layer ▔
// ▁ Network ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Layers, right folded.
 * @param Neuron     Neuron used
 * @param NeuronLast Neuron used on the last layer
 * @param ...        Input/output vector dimensions
**/
template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t input_dim, nat_t inter_dim, nat_t... output_dim> class Layers {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(inter_dim > 0, "Invalid intermediate vector dimension");
private:
    Layer<Neuron, input_dim, inter_dim> layer; // Input layer
    Layers<Neuron, NeuronLast, inter_dim, output_dim...> layers; // Output layers
public:
    /** Neuron IDs dispatcher constructor.
     * @param offset Offset for the neuron IDs
    **/
    Layers(nat_t offset): layer(offset), layers(offset + inter_dim) {}
public:
    /** Randomize the network.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        layer.randomize(rand);
        layers.randomize(rand);
    }
    /** Compute the output vector of the network.
     * @param input  Input vector
     * @param output Output vector
    **/
    template<nat_t implicit_dim> void compute(Vector<input_dim> const& input, Vector<implicit_dim>& output) {
        Vector<inter_dim> local_output; // Local layer output vector
        layer.compute(input, local_output);
        layers.compute(local_output, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input     Input vector
     * @param expected  Expected output vector
     * @param error     Error vector (output)
     * @param eta       Correction factor
     * @param error_out <Reserved>
    **/
    template<nat_t implicit_dim> void accumulate(Vector<input_dim> const& input, Vector<implicit_dim> const& expected, Vector<implicit_dim>& error, val_t eta, Vector<input_dim>* error_out = null) {
        Vector<inter_dim> local_output;
        layer.compute(input, local_output);
        Vector<inter_dim> local_error;
        layers.accumulate(local_output, expected, error, eta, &local_error);
        layer.accumulate(input, local_error, eta, error_out);
    }
    /** Apply the accumulated correction on the weight vector of all neurons.
     * @param count Number of accumulated corrections
    **/
    void update(nat_t count) {
        layer.update(count);
        layers.update(count);
    }
public:
    /** Return the number of neurons in those layers.
     * @return Number of neurons
    **/
    static constexpr nat_t length() {
        return decltype(layer)::length() + decltype(layers)::length();
    }
    /** Return the number of layers from this layer.
     * @return #layers from this layer
    **/
    static constexpr nat_t layer_length() {
        return 1 + decltype(layers)::layer_length();
    }
    /** Return the number of weights from this layer.
     * @return Number of weights (does not count bias)
    **/
    static constexpr nat_t weight_length() {
        return decltype(layer)::weight_length() + decltype(layers)::weight_length();
    }
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    static constexpr size_t size() {
        return decltype(layer)::size() + decltype(layers)::size();
    }
    /** Return the layer ID from a given neuron ID.
     * @param id Neuron ID (undefined behavior if the neuron does not belong to the network)
     * @return Layer ID in which lives the given neuron
    **/
    static constexpr nat_t neuron_to_layer(nat_t id) {
        return (id < inter_dim ? 0 : neuron_to_layer(id - inter_dim) + 1);
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        layer.load(input);
        layers.load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) const {
        layer.store(output);
        layers.store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) const {
        layer.print(ostr);
        ostr << ", ";
        layers.print(ostr);
    }
};
template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t input_dim, nat_t output_dim> class Layers<Neuron, NeuronLast, input_dim, output_dim> {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    Layer<NeuronLast, input_dim, output_dim> layer; // Input/output last layer, using the specific neuron
public:
    /** Neuron IDs dispatcher constructor.
     * @param offset Offset for the neuron IDs
    **/
    Layers(nat_t offset): layer(offset) {}
public:
    /** Randomize the network.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        layer.randomize(rand);
    }
    /** Compute the output vector of the network.
     * @param input  Input vector
     * @param output Output vector
    **/
    void compute(Vector<input_dim> const& input, Vector<output_dim>& output) {
        layer.compute(input, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input     Input vector
     * @param expected  Expected output vector
     * @param error     Error vector (output)
     * @param eta       Correction factor
     * @param error_out <Reserved>
    **/
    void accumulate(Vector<input_dim> const& input, Vector<output_dim> const& expected, Vector<output_dim>& error, val_t eta, Vector<input_dim>* error_out = null) {
        Vector<output_dim> local_output;
        layer.compute(input, local_output);
        for (nat_t i = 0; i < output_dim; i++)
            error.set(i, expected.get(i) - local_output.get(i));
        layer.accumulate(input, error, eta, error_out);
    }
    /** Apply the accumulated correction on the weight vector of all neurons.
     * @param count Number of accumulated corrections
    **/
    void update(nat_t count) {
        layer.update(count);
    }
public:
    /** Return the number of neurons in this layer.
     * @return Number of neurons
    **/
    static constexpr nat_t length() {
        return decltype(layer)::length();
    }
    /** Return the number of layers in this layer, so 1.
     * @return 1
    **/
    static constexpr nat_t layer_length() {
        return 1;
    }
    /** Return the number of weights in this layer.
     * @return Number of weights (does not count bias)
    **/
    static constexpr nat_t weight_length() {
        return decltype(layer)::weight_length();
    }
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    static constexpr size_t size() {
        return decltype(layer)::size();
    }
    /** Return the layer ID from a given neuron ID.
     * @param id Neuron ID (undefined behavior if the neuron does not belong to the network)
     * @return 0 (there is no more layer)
    **/
    static constexpr nat_t neuron_to_layer(nat_t id) {
        return 0;
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        layer.load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) const {
        layer.store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) const {
        layer.print(ostr);
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Feed-forward neural network.
 * @param Neuron     Neuron used
 * @param NeuronLast Neuron used on the last layer
 * @param ...        Input/output vector dimensions
**/
template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... dimensions> class Network final: public Layers<Neuron, NeuronLast, dimensions...> {
public:
    /** Neuron IDs dispatcher constructor.
    **/
    Network(): Layers<Neuron, NeuronLast, dimensions...>(0) {}
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Network ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
