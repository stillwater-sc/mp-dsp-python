#pragma once
// types.hpp: arithmetic configuration dispatch for Python bindings
//
// Maps runtime string keys to compile-time template instantiations.
// This avoids combinatorial explosion: 8 configs, not 216.

#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<bit>)
#include <bit>  // required by Universal on MSVC
#endif

// Universal number types
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/integer/integer.hpp>
#include <sw/universal/number/posit/posit.hpp>

namespace mpdsp {

// Arithmetic type aliases used across bindings
using cf24  = sw::universal::cfloat<24, 5, uint32_t, true, false, false>;
using half_ = sw::universal::cfloat<16, 5, uint16_t, true, false, false>;
using p32   = sw::universal::posit<32, 2>;
using p16   = sw::universal::posit<16, 1>;

// Sensor / FPGA sample-path scalars (issue #55). integer<N> is the ADC
// quantization target for sensor_* configs; fixpnt is the state/sample
// scalar pair for a fixed-point FPGA datapath.
using int8_sample_t  = sw::universal::integer<8>;
using int6_sample_t  = sw::universal::integer<6>;
using fx3224_t       = sw::universal::fixpnt<32, 24>;  // FPGA coefficient/state
using fx1612_t       = sw::universal::fixpnt<16, 12>;  // FPGA sample path

// The set of pre-instantiated arithmetic configurations.
// Python passes these as string keys; C++ dispatches to the correct types.
enum class ArithConfig {
	reference,      // double / double / double
	gpu_baseline,   // double / float / float
	ml_hw,          // double / float / half
	posit_full,     // double / posit<32,2> / posit<16,1>
	tiny_posit,     // double / posit<8,2> / posit<8,2>
	cf24_config,    // double / cf24 / cf24
	half_config,    // double / half / half
	sensor_8bit,    // double / double / integer<8>
	sensor_6bit,    // double / double / integer<6>
	fpga_fixed,     // double / fixpnt<32,24> / fixpnt<16,12>
};

inline ArithConfig parse_config(const std::string& name) {
	if (name == "reference" || name == "double") return ArithConfig::reference;
	if (name == "gpu_baseline" || name == "float") return ArithConfig::gpu_baseline;
	if (name == "ml_hw" || name == "bfloat16") return ArithConfig::ml_hw;
	if (name == "posit_full") return ArithConfig::posit_full;
	if (name == "tiny_posit") return ArithConfig::tiny_posit;
	if (name == "cf24") return ArithConfig::cf24_config;
	if (name == "half") return ArithConfig::half_config;
	if (name == "sensor_8bit") return ArithConfig::sensor_8bit;
	if (name == "sensor_6bit") return ArithConfig::sensor_6bit;
	if (name == "fpga_fixed") return ArithConfig::fpga_fixed;
	throw std::invalid_argument("Unknown arithmetic config: " + name);
}

// List of available config names (for Python introspection)
inline std::vector<std::string> available_configs() {
	return { "reference", "gpu_baseline", "ml_hw", "posit_full",
	         "tiny_posit", "cf24", "half",
	         "sensor_8bit", "sensor_6bit", "fpga_fixed" };
}

// Sample-scalar bit width per config. Surfaces the table that was
// previously hardcoded in scripts/plot_dashboard.py:plot_precision_cost_frontier
// so the dashboard and any other consumer can query it rather than duplicate.
// Returns the width of the narrowest (sample-path) scalar, which is what the
// precision-vs-cost frontier is plotted against.
inline int bits_of(const std::string& name) {
	auto cfg = parse_config(name);
	switch (cfg) {
	case ArithConfig::reference:    return 64;
	case ArithConfig::gpu_baseline: return 32;
	case ArithConfig::ml_hw:        return 16;  // half
	case ArithConfig::posit_full:   return 16;  // posit<16,1> sample path
	case ArithConfig::tiny_posit:   return 8;   // posit<8,2>
	case ArithConfig::cf24_config:  return 24;
	case ArithConfig::half_config:  return 16;
	case ArithConfig::sensor_8bit:  return 8;
	case ArithConfig::sensor_6bit:  return 6;
	case ArithConfig::fpga_fixed:   return 16;  // fixpnt<16,12> sample path
	}
	throw std::invalid_argument("bits_of: unsupported ArithConfig");
}

} // namespace mpdsp
