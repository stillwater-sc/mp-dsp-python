#pragma once
// types.hpp: arithmetic configuration dispatch for Python bindings
//
// Maps runtime string keys to compile-time template instantiations.
// This avoids combinatorial explosion: N configs, not NNN.

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

// Non-posit type aliases used across bindings
using cf24  = sw::universal::cfloat<24, 5, uint32_t, true, false, false>;
using half_ = sw::universal::cfloat<16, 5, uint16_t, true, false, false>;

// Posit taxonomy grid (issue #81) — posit<N, es> for every combination
// of N ∈ {8, 16, 32} and es ∈ {0, 1, 2}. Single-type configs: coefficient,
// state, and sample all use the same posit type. The mixed `posit_full`
// bundle (posit<32,2> state + posit<16,1> sample) is orthogonal and
// unchanged — it represents a specific production pipeline, not a grid
// cell.
using p8_0  = sw::universal::posit<8, 0>;
using p8_1  = sw::universal::posit<8, 1>;
using p8_2  = sw::universal::posit<8, 2>;
using p16_0 = sw::universal::posit<16, 0>;
using p16_1 = sw::universal::posit<16, 1>;
using p16_2 = sw::universal::posit<16, 2>;
using p32_0 = sw::universal::posit<32, 0>;
using p32_1 = sw::universal::posit<32, 1>;
using p32_2 = sw::universal::posit<32, 2>;

// Legacy aliases from before the taxonomy grid landed. `p32` and `p16`
// were named for their roles in the posit_full bundle (32-bit state +
// 16-bit sample); they live on as aliases for the corresponding grid
// cells so existing dispatcher code paths keep compiling.
using p32 = p32_2;
using p16 = p16_1;

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
	cf24_config,    // double / cf24 / cf24
	half_config,    // double / half / half
	sensor_8bit,    // double / double / integer<8>
	sensor_6bit,    // double / double / integer<6>
	fpga_fixed,     // double / fixpnt<32,24> / fixpnt<16,12>
	// Posit taxonomy grid (#81) — 9 cells (N × es).
	posit_8_0,      // posit<8,0>  throughout
	posit_8_1,      // posit<8,1>  throughout
	posit_8_2,      // posit<8,2>  throughout
	posit_16_0,     // posit<16,0> throughout
	posit_16_1,     // posit<16,1> throughout
	posit_16_2,     // posit<16,2> throughout
	posit_32_0,     // posit<32,0> throughout
	posit_32_1,     // posit<32,1> throughout
	posit_32_2,     // posit<32,2> throughout
	// Legacy alias for the pre-#81 "tiny_posit" name. Same underlying
	// enum value as posit_8_2, so switches only need the posit_8_2 case
	// (adding a tiny_posit case would be a duplicate-case compile error).
	// parse_config accepts "tiny_posit" as input; available_configs()
	// does not list it — the taxonomic name is canonical.
	tiny_posit = posit_8_2,
};

inline ArithConfig parse_config(const std::string& name) {
	if (name == "reference" || name == "double") return ArithConfig::reference;
	if (name == "gpu_baseline" || name == "float") return ArithConfig::gpu_baseline;
	if (name == "ml_hw" || name == "bfloat16") return ArithConfig::ml_hw;
	if (name == "posit_full") return ArithConfig::posit_full;
	if (name == "cf24") return ArithConfig::cf24_config;
	if (name == "half") return ArithConfig::half_config;
	if (name == "sensor_8bit") return ArithConfig::sensor_8bit;
	if (name == "sensor_6bit") return ArithConfig::sensor_6bit;
	if (name == "fpga_fixed") return ArithConfig::fpga_fixed;
	// Posit taxonomy grid (#81).
	if (name == "posit_8_0")  return ArithConfig::posit_8_0;
	if (name == "posit_8_1")  return ArithConfig::posit_8_1;
	if (name == "posit_8_2")  return ArithConfig::posit_8_2;
	if (name == "posit_16_0") return ArithConfig::posit_16_0;
	if (name == "posit_16_1") return ArithConfig::posit_16_1;
	if (name == "posit_16_2") return ArithConfig::posit_16_2;
	if (name == "posit_32_0") return ArithConfig::posit_32_0;
	if (name == "posit_32_1") return ArithConfig::posit_32_1;
	if (name == "posit_32_2") return ArithConfig::posit_32_2;
	// Legacy alias — maps to the same enum value as posit_8_2. Kept so
	// any caller still passing "tiny_posit" continues to work. Remove
	// if a major-version cleanup ever drops it; not listed in
	// available_configs() already.
	if (name == "tiny_posit") return ArithConfig::posit_8_2;
	throw std::invalid_argument("Unknown arithmetic config: " + name);
}

// List of available config names (for Python introspection). The
// "tiny_posit" alias is intentionally absent — parse_config still
// accepts it, but the canonical name for that config is "posit_8_2".
inline std::vector<std::string> available_configs() {
	return { "reference", "gpu_baseline", "ml_hw", "posit_full",
	         "cf24", "half",
	         "sensor_8bit", "sensor_6bit", "fpga_fixed",
	         "posit_8_0", "posit_8_1", "posit_8_2",
	         "posit_16_0", "posit_16_1", "posit_16_2",
	         "posit_32_0", "posit_32_1", "posit_32_2" };
}

// Sample-scalar bit width per config. Surfaces the table that was
// previously hardcoded in scripts/plot_dashboard.py:plot_precision_cost_frontier
// so the dashboard and any other consumer can query it rather than duplicate.
// Returns the width of the narrowest (sample-path) scalar, which is what the
// precision-vs-cost frontier is plotted against. For posit grid cells, the
// ES dimension doesn't affect bit width — only exponent encoding — so every
// posit_N_* config reports N.
inline int bits_of(const std::string& name) {
	auto cfg = parse_config(name);
	switch (cfg) {
	case ArithConfig::reference:    return 64;
	case ArithConfig::gpu_baseline: return 32;
	case ArithConfig::ml_hw:        return 16;  // half
	case ArithConfig::posit_full:   return 16;  // posit<16,1> sample path
	case ArithConfig::cf24_config:  return 24;
	case ArithConfig::half_config:  return 16;
	case ArithConfig::sensor_8bit:  return 8;
	case ArithConfig::sensor_6bit:  return 6;
	case ArithConfig::fpga_fixed:   return 16;  // fixpnt<16,12> sample path
	case ArithConfig::posit_8_0:    return 8;
	case ArithConfig::posit_8_1:    return 8;
	case ArithConfig::posit_8_2:    return 8;   // also covers tiny_posit alias
	case ArithConfig::posit_16_0:   return 16;
	case ArithConfig::posit_16_1:   return 16;
	case ArithConfig::posit_16_2:   return 16;
	case ArithConfig::posit_32_0:   return 32;
	case ArithConfig::posit_32_1:   return 32;
	case ArithConfig::posit_32_2:   return 32;
	}
	throw std::invalid_argument("bits_of: unsupported ArithConfig");
}

} // namespace mpdsp
