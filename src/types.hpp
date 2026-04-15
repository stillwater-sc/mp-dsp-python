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
#include <sw/universal/number/posit/posit.hpp>

namespace mpdsp {

// Arithmetic type aliases used across bindings
using cf24  = sw::universal::cfloat<24, 5, uint32_t, true, false, false>;
using half_ = sw::universal::cfloat<16, 5, uint16_t, true, false, false>;
using p32   = sw::universal::posit<32, 2>;
using p16   = sw::universal::posit<16, 1>;

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
};

inline ArithConfig parse_config(const std::string& name) {
	if (name == "reference" || name == "double") return ArithConfig::reference;
	if (name == "gpu_baseline" || name == "float") return ArithConfig::gpu_baseline;
	if (name == "ml_hw" || name == "bfloat16") return ArithConfig::ml_hw;
	if (name == "posit_full") return ArithConfig::posit_full;
	if (name == "tiny_posit") return ArithConfig::tiny_posit;
	if (name == "cf24") return ArithConfig::cf24_config;
	if (name == "half") return ArithConfig::half_config;
	throw std::invalid_argument("Unknown arithmetic config: " + name);
}

// List of available config names (for Python introspection)
inline std::vector<std::string> available_configs() {
	return { "reference", "gpu_baseline", "ml_hw", "posit_full",
	         "tiny_posit", "cf24", "half" };
}

} // namespace mpdsp
