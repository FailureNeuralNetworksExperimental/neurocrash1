/**
 * @file mnist.cpp
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
#define dimensions input_dim, first_layer, second_layer, third_layer, output_dim // Network dimensions
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
static constexpr nat_t rows_length  = 28; // Image row length
static constexpr nat_t cols_length  = 28; // Image col length
static constexpr nat_t input_dim    = rows_length * cols_length; // Input space dimension
static constexpr nat_t output_dim   = 10; // Output space dimension
static constexpr nat_t first_layer  = 20; // First hidden layer output dimension
static constexpr nat_t second_layer = 12; // Second hidden layer output dimension
static constexpr nat_t third_layer  = 11; // Third hidden layer output dimension
static constexpr nat_t robust_killfrom = robust_kf_val; // Offset in the killmap
static constexpr nat_t robust_killto   = robust_kt_val; // Length kill from the offset in the killmap
static val_t const value_valid    = 1;   // Value for "valid dimension"
static val_t const value_invalid  = 0;   // Value for "invalid dimension"
static val_t const margin_valid   = 0.4; // Margin for "valid dimension"
static val_t const margin_invalid = 0.4; // Margin for "invalid dimensions"
static val_t const base_eta = 0.5; // Base learning rate
static nat_t const epoch_limit = 10000; // Max number of epochs
static nat_t const count_limit = 5000;  // Max number of ill-classifications for the network to be considered valid

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
template<nat_t input_dim> using HiddenNeuron = StandardNeuron<HiddenTransfert>::Neuron<input_dim>;

/** Output transfert neuron.
 * @param input_dim Input dimension
**/
template<nat_t input_dim> using OutputNeuron = StandardNeuron<LinearTransfert>::Neuron<input_dim>;

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

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Return the label associated with a dimension id.
 * @param dim Given dimension id
 * @return Associated label
**/
constexpr nat_t dim_to_label(nat_t dim) {
    return dim;
}

/** Return the dimension id associated with a label.
 * @param label Given label
 * @return Associated dimension id
**/
constexpr nat_t label_to_dim(nat_t label) {
    return label;
}

/** Initialize an output, and optionaly a margin vector from a label.
 * @param label  Label to translate
 * @param output Output vector
 * @param margin Margin vector (optional)
**/
void label_to_vector(nat_t label, Output& output, Output* margin = null) {
    nat_t dim_label = label_to_dim(label);
    for (nat_t i = 0; i < output_dim; i++)
        output.set(i, i == dim_label ? value_valid : value_invalid);
    if (margin)
        for (nat_t i = 0; i < output_dim; i++)
            margin->set(i, i == dim_label ? margin_valid : margin_invalid);
}

/** Translate an output vector to a label.
 * @param output Output vector to translate
 * @return Associated label
**/
nat_t vector_to_label(Output& output) {
    nat_t largest_dim = 0;
    val_t largest_val = output.get(0);
    for (nat_t i = 1; i < output_dim; i++) {
        val_t val = output.get(i);
        if (val > largest_val) {
            largest_dim = i;
            largest_val = val;
        }
    }
    return dim_to_label(largest_dim);
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
// ▁ Database ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Data parser from files.
**/
class Loader final {
private:
    /** File descriptor.
    **/
    using File = int;
    /** File map.
    **/
    class Map final {
    private:
        uint8_t* data;   // Data pages, null if none
        size_t   length; // Mapping length, in bytes
        size_t   cursor; // Cursor on the mapping
    public:
        /** Build a map to a file.
         * @param path Path to the file to map
        **/
        Map(char const* path) {
            File fd = open(path, O_RDONLY);
            if (fd == -1) {
                ::std::string err_str;
                err_str.append("Unable to open '");
                err_str.append(path);
                err_str.append("' for reading");
                throw ::std::runtime_error(err_str);
            }
            length = lseek(fd, 0, SEEK_END); // Get file size, in bytes
            data = static_cast<uint8_t*>(mmap(null, length, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
            if (data == MAP_FAILED)
                throw ::std::runtime_error("Mapping failed");
            close(fd);
            cursor = 0;
        }
        /** Destroy a map.
        **/
        ~Map() {
            if (likely(data))
                munmap(data, length);
        }
    public:
        /** Return a part of the data as an object of a given class, move the cursor to the next object.
         * @param Type Type of the returned object
         * @return Pointer to the object
        **/
        template<class Type> Type* read() {
            static_assert(!::std::is_polymorphic<Type>::value, "Given class is polymorphic");
            Type* ret = static_cast<Type*>(static_cast<void*>(data + cursor));
            cursor += sizeof(Type);
            if (cursor > length)
                throw ::std::runtime_error("Read out of file bounds");
            return ret;
        }
        /** Only move the cursor for a given number of bytes.
         * @param delta Number of bytes
        **/
        void seek(size_t delta) {
            cursor += delta;
        }
    };
    /** Image entry.
     * @param dim Entry total size
    **/
    class Entry final {
    private:
        uint8_t data[input_dim];
    private:
        /** Convert a grey-scale to an input level.
         * @param color Grey-scale to convert
         * @return Input level (+1 white ... 0 black)
        **/
        val_t convert(nat_t color) const {
            return static_cast<val_t>(color) / 255;
        }
    public:
        /** Initialize a vector with such data.
         * @param vector Vector to initialize
        **/
        void dump(Input& vector) const {
            for (nat_t i = 0; i < input_dim; i++)
                vector.set(i, convert(data[i]));
        }
    };
private:
    Map img; // File descriptor (-1 for none) for the images file
    Map lab; // File descriptor (-1 for none) for the labels file
    nat_t count; // Remaining images
private:
    /** Inverse endianess.
     * @param UInt  Implicit unsigned integer type
     * @param value Value to inverse
     * @return Value inversed
    **/
    template<class Uint> Uint endian_inverse(Uint value) {
        Uint ret;
        uint8_t* v = static_cast<uint8_t*>(static_cast<void*>(&value));
        uint8_t* r = static_cast<uint8_t*>(static_cast<void*>(&ret));
        for (nat_t i = 0; i < sizeof(Uint); i++)
            r[i] = v[sizeof(Uint) - 1 - i];
        return ret;
    }
public:
    /** Open images/labels files, basic validity checks.
     * @param path_img Images file to open
     * @param path_lab Labels file to open
    **/
    Loader(char const* path_img, char const* path_lab): img(path_img), lab(path_lab) {
        uint32_t (&header_img)[4] = *img.read<uint32_t[4]>(); // Magic number, image count, row size, column size
        uint32_t (&header_lab)[2] = *lab.read<uint32_t[2]>(); // Magic number, label count
        if (header_img[2] != endian_inverse<uint32_t>(rows_length) || header_img[3] != endian_inverse<uint32_t>(cols_length)) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' invalid dimensions");
            throw ::std::runtime_error(err_str);
        }
        if (header_img[1] != header_lab[1]) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' and '");
            err_str.append(path_lab);
            err_str.append("' count mismatch");
            throw ::std::runtime_error(err_str);
        }
        count = static_cast<nat_t>(endian_inverse(header_img[1]));
        if (unlikely(count < 1)) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' and '");
            err_str.append(path_lab);
            err_str.append("' no image");
            throw ::std::runtime_error(err_str);
        }
    }
public:
    /** Initialize an input vector and an associated label.
     * @param vector Input vector
     * @param label  Associated label
     * @return True if another vector/label exists, false otherwise
    **/
    bool feed(Input& vector, nat_t& label) {
        if (unlikely(count == 0))
            throw ::std::runtime_error("No more image to feed");
        Entry& image = *img.read<Entry>();
        uint8_t& lbl = *lab.read<uint8_t>();
        image.dump(vector);
        label = static_cast<nat_t>(lbl);
        return --count != 0;
    }
};

/** Tests set.
**/
class Tests final {
private:
    /** A labelled test image.
    **/
    class Image final {
    private:
        /** PGM binary data (256 colors) output serializer.
        **/
        class PGM final: public Serializer::Output {
        private:
            ::std::ofstream& file; // Output file
        public:
            /** File initializer.
             * @param file File stream to initialize with
            **/
            PGM(::std::ofstream& file): file(file) {}
        public:
            /** Store a pixel to the file.
             * @param level Color level (-1 white ... +1 black)
            **/
            virtual void store(val_t value) {
                uint8_t pixel;
                value = val_t(255) - (value + val_t(1)) * val_t(128);
                if (value < 1) {
                    pixel = 0;
                } else if (value > 254) {
                    pixel = 255;
                } else {
                    pixel = static_cast<uint8_t>(value);
                }
                file.write(reinterpret_cast<::std::remove_reference<decltype(file)>::type::char_type*>(&pixel), sizeof(uint8_t));
            }
        };
    public:
        Input image; // Associated input vector
        nat_t label; // Number represented
    public:
        /** Check if the network answered correctly.
         * @param network Network to test
         * @param guess   Guess made
         * @return True on a correct answer, false otherwise
        **/
        template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> bool check(Network<Neuron, NeuronLast, implicit_dims...>& network, nat_t& guess) const {
            Output result;
            network.compute(image, result);
            Output softmax = result.softmax();
            guess = Helper::vector_to_label(softmax);
            return guess == label;
        }
        /** Output the picture to the given file, overwrite the file.
         * @param filename File to write
        **/
        void output(::std::string& filename) const {
            ::std::ofstream file(filename);
            file << "P5\n" << ::std::to_string(rows_length) << " " << ::std::to_string(cols_length) << " 255\n";
            PGM serializer(file);
            image.store(serializer);
        }
    };
private:
    ::std::vector<Image> tests; // List of test images with labels
public:
    /** Load images and labels from loader object.
     * @param loader Loader object to load from
    **/
    void load(Loader& loader) {
        while (true) { // At least one element in loader
            tests.emplace(tests.end());
            Image& current = tests.back(); // Current picture
            if (!loader.feed(current.image, current.label))
                break;
        }
    }
    /** Test network on the testing set.
     * @param network  Network to test
     * @param errordir Directory to which failed test image are output (optional, null for no output)
     * @return Number of success, number of test elements
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> ::std::tuple<nat_t, nat_t> test(Network<Neuron, NeuronLast, implicit_dims...>& network, char const* const errordir = null) const {
        nat_t count = 0; // Success counter
        nat_t error = 0; // Error counter
        for (Image const& test: tests) {
            nat_t guess;
            if (test.check(network, guess)) {
                count++;
            } else if (errordir) {
                ::std::string filename = ::std::string(errordir) + "/" + ::std::to_string(error++) + "_guessed_" + ::std::to_string(guess) + "_for_" + ::std::to_string(test.label) + ".pgm";
                test.output(filename);
            }
        }
        return ::std::make_tuple(count, static_cast<nat_t>(tests.size()));
    }
    /** Compute the epsilon value of the network.
     * @param network Network to use
     * @return Value of max epsilon, value of average epsilon on the whole test set
    **/
    /*template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, nat_t... implicit_dims> ::std::tuple<val_t, val_t, val_t> epsilon(Network<Neuron, NeuronLast, implicit_dims...>& network) const {
        val_t max_epsilon = 0;
        val_t avg_epsilon = 0;
        val_t std_epsilon = 0;
        val_t size = tests.size();
        for (Image const& test: tests) { // For each image in the test set
            Output result;
            Output expected;
            network.compute(test.image, result);
            Helper::label_to_vector(test.label, expected);
            result -= expected;
            val_t epsilon = result.norm(); // Infinite norm
            if (epsilon > max_epsilon)
                max_epsilon = epsilon;
            avg_epsilon += epsilon / size;
            std_epsilon += (epsilon * epsilon) / size;
        }
        return ::std::make_tuple(max_epsilon, avg_epsilon, std_epsilon);
    }*/
    /** Compute the omega value of the network.
     * @param network   Network to use
     * @param reference Network reference to use
     * @return Value of max omega, value of average omega on the whole test set
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, template<nat_t> class NeuronRef, template<nat_t> class NeuronRefLast, nat_t... implicit_dims> ::std::tuple<val_t, val_t, val_t> omega(Network<Neuron, NeuronLast, implicit_dims...>& network, Network<NeuronRef, NeuronRefLast, implicit_dims...>& reference) const {
        val_t max_omega = 0;
        val_t avg_omega = 0;
        val_t std_omega = 0;
        val_t size = tests.size();
        for (Image const& test: tests) { // For each image in the test set
            Output result;
            Output expected;
            network.compute(test.image, result);
            reference.compute(test.image, expected);
            Output soft_result = result.softmax(); // Actually compute the softmax of outputs
            Output soft_expect = expected.softmax();
            soft_result -= soft_expect;
            val_t omega = soft_result.norm(); // Infinite norm
            if (omega > max_omega)
                max_omega = omega;
            avg_omega += omega / size;
            std_omega += (omega * omega) / size;
        }
        return ::std::make_tuple(max_omega, avg_omega, std_omega);
    }
    /** Compute robustness-related metrics (epsilon and omega).
     * @param network   Network to use
     * @param reference Network reference for omega computation
    **/
    template<template<nat_t> class Neuron, template<nat_t> class NeuronLast, template<nat_t> class NeuronRef, template<nat_t> class NeuronRefLast, nat_t... implicit_dims> void robust(Network<Neuron, NeuronLast, implicit_dims...>& network, Network<NeuronRef, NeuronRefLast, implicit_dims...>& reference) const {
        nat_t const offset_limit = robust_killto; // Amount of killable neurons
        for (nat_t i = 0; i < NetworkDefault::length(); i++) // Reset the entire map
            killmap[i] = false;
        bool* killview = killmap + robust_killfrom; // Set of neurons potentially killed
        for (nat_t i = 0; i <= offset_limit; i++) {
            // val_t epsilon_max = 0; // Max epsilon
            // val_t epsilon_avg = 0; // Avg epsilon
            // val_t epsilon_std = 0; // Std dev epsilon
            val_t omega_max = 0; // Max omega
            val_t omega_avg = 0; // Avg omega
            val_t omega_std = 0; // Std dev omega
            for (nat_t j = 0; j < offset_limit; j++) // Initialize the array
                killview[j] = (j < i ? true : false);
            nat_t combs = Helper::combination(offset_limit, i);
            for (nat_t j = 0;;) { // For all possible combinations
                /*{ // "Partially compute" epsilon_max and epsilon_avg
                    val_t test_max, test_avg, test_std;
                    ::std::tie(test_max, test_avg, test_std) = epsilon(network);
                    if (test_max > epsilon_max)
                        epsilon_max = test_max;
                    epsilon_avg += test_avg / static_cast<val_t>(combs);
                    epsilon_std += test_std / static_cast<val_t>(combs);
                }*/
                if (i != 0 || j != 0) { // "Partially compute" omega_max and omega_avg (avoid case where no neuron is killed)
                    val_t test_max, test_avg, test_std;
                    ::std::tie(test_max, test_avg, test_std) = omega(network, reference);
                    if (test_max > omega_max)
                        omega_max = test_max;
                    omega_avg += test_avg / static_cast<val_t>(combs);
                    omega_std += test_std / static_cast<val_t>(combs);
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
            // epsilon_std = ::std::sqrt(epsilon_std - epsilon_avg * epsilon_avg);
            omega_std = ::std::sqrt(omega_std - omega_avg * omega_avg);
            // ::std::cout << i << "\t" << epsilon_max << "\t" << epsilon_avg << "\t" << epsilon_std << "\t" << omega_max << "\t" << omega_avg << "\t" << omega_std << ::std::endl;
            ::std::cout << i << "\t" << omega_max << "\t" << omega_avg << "\t" << omega_std << ::std::endl;
        }
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Learning discipline used to train networks
static Learning<input_dim, output_dim> discipline;

// Tests set
static Tests tests;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Database ▔
// ▁ Orders ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Learning order handler.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int train(int argc, char** argv) {
    if (argc < 5 || argc > 6) { // Wrong number of parameters
        ::std::cerr << "Usage: " << argv[0] << " " << argv[1] << " <training images> <training labels> <k-param> [batch size] | 'raw trained network'" << ::std::endl;
        return 0;
    }
    nat_t batch_size; // Batch size
    if (argc >= 6) { // Init batch_size
        long long const bs = ::std::atoll(argv[5]);
        if (bs <= 0) {
            ::std::cerr << "The batch size must be a positive integer" << ::std::endl;
            return 1;
        }
        batch_size = static_cast<nat_t>(bs);
    } else {
        batch_size = 1;
    }
    if (!transfert_init(::std::atof(argv[4]))) // Initialize transfert functions
        return 1;
    { // Loading phase
        ::std::cerr << "Loading training files...";
        ::std::cerr.flush();
        try {
            Loader train(argv[2], argv[3]);
            Input input;
            nat_t label;
            while (true) {
                bool cont = train.feed(input, label); // There is at least one element to feed
                Output output;
                Output margin;
                Helper::label_to_vector(label, output, &margin);
                discipline.add(input, output, margin);
                if (!cont)
                    break;
            }
        } catch (::std::runtime_error& err) {
            ::std::cerr << " fail: " << err.what() << ::std::endl;
            return 1;
        }
        ::std::cerr << " done." << ::std::endl;
    }
    NetworkDefault network; // Using default network
    { // Randomize network
        UniformRandomizer randomizer(0.01);
        network.randomize(randomizer);
    }
    { // Learning phase
        static bool volatile run = true; // Should be static to be modified by the interrupt handler
        { // Output network on interrupt signal
            class Local final {
            public:
                /** Make the loop stops on interrupt signal.
                 * @param Signal received (ignored, should be SIGINT)
                **/
                static void stop(int) {
                    run = false;
                }
            };
            signal(SIGINT, Local::stop);
        }
        ::std::cerr << "Learning phase...";
        ::std::cerr.flush();
        discipline.batch_size(batch_size);
        nat_t epoch = 0;
        val_t eta = base_eta;
        while (likely(run && epoch < epoch_limit)) {
            nat_t count = discipline.correct(network, eta);
            ::std::cerr << "\rLearning phase... eta " << eta << " epoch " << ++epoch << ": " << count << "          ";
            if (count <= count_limit) // Stop when enough get well classified
                break;
            ::std::cerr.flush();
            discipline.shuffle();
            if (epoch < 4000) {
                eta = (4000. / 98.) / ((8000. / 98.) + static_cast<val_t>(epoch));
            } else if (epoch == 4000) {
                eta = 0.01;
            }
        }
        { // Remove interrupt signal handling
            signal(SIGINT, SIG_DFL);
        }
        ::std::cerr << "\rLearning phase... final epoch " << epoch << " done.          " << ::std::endl;
    }
    { // Output phase
        Serializer::StreamOutput so(::std::cout);
        network.store(so);
    }
    return 0;
}

/** Test order handler.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int test(int argc, char** argv) {
    if (argc < 5 || argc > 6) { // Wrong number of parameters
        ::std::cerr << "Usage: 'raw trained network' | " << argv[0] << " " << argv[1]  << " <test images> <test labels> <k-param> [path/to/error/directory]" << ::std::endl;
        return 0;
    }
    if (!transfert_init(::std::atof(argv[4]))) // Initialize transfert functions
        return 1;
    { // Loading phase
        ::std::cerr << "Loading testing files...";
        ::std::cerr.flush();
        try {
            Loader test(argv[2], argv[3]);
            tests.load(test);
        } catch (::std::runtime_error& err) {
            ::std::cerr << " fail: " << err.what() << ::std::endl;
            return 1;
        }
        ::std::cerr << " done." << ::std::endl;
    }
    NetworkDefault network; // Default network
    { // Input phase
        Serializer::StreamInput si(::std::cin);
        network.load(si);
    }
    { // Testing phase
        ::std::cerr << "Testing phase...";
        ::std::cerr.flush();
        nat_t success;
        nat_t total;
        ::std::tie(success, total) = tests.test(network, (argc == 6 ? argv[5] : null));
        ::std::cerr << " " << success << "/" << total << ::std::endl;
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

/** Generate a random network.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int rand(int argc, char** argv) {
    if (argc < 2 || argc > 3) { // Wrong number of parameters
        ::std::cerr << "Usage: " << argv[0] << " " << argv[1] << " [limit] | 'raw random network'" << ::std::endl;
        return 0;
    }
    NetworkDefault network; // Default network
    UniformRandomizer randomizer(argc == 3 ? ::std::atof(argv[2]) : 0.01);
    network.randomize(randomizer);
    Serializer::StreamOutput so(::std::cout);
    network.store(so);
    return 0;
}

/** Robustness-related measurements.
 * @param argc Number of arguments
 * @param argv Arguments (at least 2)
 * @return Return code
**/
int robust(int argc, char** argv) {
    if (argc != 5) { // Wrong number of parameters
        ::std::cerr << "Usage: 'raw trained network' | " << argv[0] << " " << argv[1]  << " <test images> <test labels> <k-param> | 'robustness data'" << ::std::endl;
        return 0;
    }
    if (!transfert_init(::std::atof(argv[4]))) // Initialize transfert functions
        return 1;
    { // Loading phase
        ::std::cerr << "Loading testing files...";
        ::std::cerr.flush();
        try {
            Loader test(argv[2], argv[3]);
            tests.load(test);
        } catch (::std::runtime_error& err) {
            ::std::cerr << " fail: " << err.what() << ::std::endl;
            return 1;
        }
        ::std::cerr << " done." << ::std::endl;
    }
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
        tests.robust(network, reference);
        ::std::cerr << " done." << ::std::endl;
    }
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
    { "train", train },
    { "test", test },
    { "robust", robust },
    { "plot", plot },
    { "rand", rand },
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
