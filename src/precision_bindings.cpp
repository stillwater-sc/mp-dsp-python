// precision_bindings.cpp: precision-vs-magnitude profiles ("decimals of
// accuracy") for choosing a number system by where it spends its bits.
//
// Issue #133. Binds Universal's `utility/precision_profile.hpp` (Universal
// #1650, v5.2.0+) so the decimals-of-accuracy tutorial
// (universal/docs/tutorials/decimals-of-accuracy.md) can be reproduced as a
// single Python/matplotlib workflow instead of a C++ application writing
// CSVs for a separate plotting script.
//
// Profile types get their own string-keyed table rather than new
// ArithConfig enumerators. Profiling a type instantiates one cheap header;
// an ArithConfig instantiates every DSP module for that type. The tutorial
// needs 17 types, most of which no algorithm here is dispatched on, so
// keeping them out of the dispatch model holds the instantiation count
// down. The ArithConfig scalars are in the table too, so a profile can be
// laid over the dtypes the DSP algorithms actually run in.
//
// Keys match the labels Universal's `dsp_precision_profiles` application
// uses, so profiles, tables and plots line up with the tutorial verbatim.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sw/universal/number/bposit/bposit.hpp>
#include <sw/universal/number/lns/lns.hpp>
#include <sw/universal/utility/precision_profile.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

namespace nb = nanobind;

namespace {

using mpdsp::bindings::np_f64;
using mpdsp::bindings::make_f64_array;
using sw::universal::precision_profile;
using sw::universal::precision_point;

using Builder = std::function<precision_profile(const std::string&, unsigned)>;

template <typename T>
Builder builder() {
	return [](const std::string& label, unsigned samples_per_binade) {
		return sw::universal::precision_profile_of<T>(label,
			samples_per_binade);
	};
}

// (key, label, builder). The label is what tables and plot legends show;
// it differs from the key only for aliases.
struct ProfileEntry {
	const char* key;
	const char* label;
	Builder build;
};

const std::vector<ProfileEntry>& profile_table() {
	using namespace sw::universal;
	static const std::vector<ProfileEntry> table = {
		// 16-bit: the tutorial's comparison set and b-posit range fits
		{ "int16",          "int16",          builder<std::int16_t>() },
		{ "Q15",            "Q15",            builder<fixpnt<16, 15, Modulo, std::uint16_t>>() },
		{ "fp16",           "fp16",           builder<mpdsp::half_>() },
		{ "half",           "fp16",           builder<mpdsp::half_>() },
		{ "lns<16,10>",     "lns<16,10>",     builder<lns<16, 10, std::uint16_t>>() },
		{ "bposit<16,6,5>", "bposit<16,6,5>", builder<bposit<16, 6, 5, std::uint16_t>>() },
		{ "bposit<16,6,3>", "bposit<16,6,3>", builder<bposit<16, 6, 3, std::uint16_t>>() },
		{ "bposit<16,6,2>", "bposit<16,6,2>", builder<bposit<16, 6, 2, std::uint16_t>>() },
		{ "bposit<16,5,2>", "bposit<16,5,2>", builder<bposit<16, 5, 2, std::uint16_t>>() },
		{ "bposit<16,4,2>", "bposit<16,4,2>", builder<bposit<16, 4, 2, std::uint16_t>>() },
		{ "bposit<16,3,1>", "bposit<16,3,1>", builder<bposit<16, 3, 1, std::uint16_t>>() },
		// 32-bit
		{ "int32",          "int32",          builder<std::int32_t>() },
		{ "Q31",            "Q31",            builder<fixpnt<32, 31, Modulo, std::uint32_t>>() },
		{ "float",          "float",          builder<float>() },
		{ "lns<32,23>",     "lns<32,23>",     builder<lns<32, 23, std::uint32_t>>() },
		{ "bposit<32,6,5>", "bposit<32,6,5>", builder<bposit<32, 6, 5, std::uint32_t>>() },
		{ "bposit<32,5,2>", "bposit<32,5,2>", builder<bposit<32, 5, 2, std::uint32_t>>() },
		{ "bposit<32,4,2>", "bposit<32,4,2>", builder<bposit<32, 4, 2, std::uint32_t>>() },
		// the remaining scalars of the ArithConfig table (types.hpp)
		{ "double",         "double",         builder<double>() },
		{ "cf24",           "cf24",           builder<mpdsp::cf24>() },
		{ "posit<8,0>",     "posit<8,0>",     builder<mpdsp::p8_0>() },
		{ "posit<8,1>",     "posit<8,1>",     builder<mpdsp::p8_1>() },
		{ "posit<8,2>",     "posit<8,2>",     builder<mpdsp::p8_2>() },
		{ "posit<16,0>",    "posit<16,0>",    builder<mpdsp::p16_0>() },
		{ "posit<16,1>",    "posit<16,1>",    builder<mpdsp::p16_1>() },
		{ "posit<16,2>",    "posit<16,2>",    builder<mpdsp::p16_2>() },
		{ "posit<32,0>",    "posit<32,0>",    builder<mpdsp::p32_0>() },
		{ "posit<32,1>",    "posit<32,1>",    builder<mpdsp::p32_1>() },
		{ "posit<32,2>",    "posit<32,2>",    builder<mpdsp::p32_2>() },
		{ "fixpnt<32,24>",  "fixpnt<32,24>",  builder<mpdsp::fx3224_t>() },
		{ "fixpnt<16,12>",  "fixpnt<16,12>",  builder<mpdsp::fx1612_t>() },
		{ "integer<8>",     "integer<8>",     builder<mpdsp::int8_sample_t>() },
		{ "integer<6>",     "integer<6>",     builder<mpdsp::int6_sample_t>() },
	};
	return table;
}

// One column of the profile as an owning float64 array.
np_f64 column(const precision_profile& p, double precision_point::*field) {
	double* out = nullptr;
	auto arr = make_f64_array(p.points.size(), out);
	for (std::size_t i = 0; i < p.points.size(); ++i) {
		out[i] = p.points[i].*field;
	}
	return arr;
}

}  // namespace

void bind_precision(nb::module_& m) {
	// Array getters build capsule-owned arrays: take_ownership is
	// required (see src/BINDING_PATTERNS.md).
	constexpr auto own = nb::rv_policy::take_ownership;

	nb::class_<precision_profile>(m, "PrecisionProfile",
			"Precision as a function of magnitude for one number system: "
			"the decimals of accuracy -log10(ulp / (2|x|)) and the fraction "
			"bits log2(|x| / ulp) at every positive encoding x (Gustafson, "
			"*The End of Error*). Types of up to 16 bits are profiled "
			"exhaustively; wider types are sampled a few points per binade. "
			"Create with `precision_profile(type)`.")
		.def_ro("label", &precision_profile::label,
		     "Short name for tables and plot legends.")
		.def_ro("type", &precision_profile::type,
		     "Universal's type_tag for the profiled type.")
		.def_ro("minpos", &precision_profile::minpos,
		     "Smallest positive value; 0 decimals below it.")
		.def_ro("maxpos", &precision_profile::maxpos,
		     "Largest positive value; 0 decimals above it.")
		.def_ro("exhaustive", &precision_profile::exhaustive,
		     "True when every positive encoding was decoded; False when "
		     "the profile is sampled per binade.")
		.def_prop_ro("magnitude", [](const precision_profile& p) {
			return column(p, &precision_point::magnitude);
		}, own, "Profiled magnitudes x, ascending.")
		.def_prop_ro("log2_magnitude", [](const precision_profile& p) {
			return column(p, &precision_point::log2_magnitude);
		}, own, "log2(x) for each profiled magnitude.")
		.def_prop_ro("decimals", [](const precision_profile& p) {
			return column(p, &precision_point::decimals);
		}, own, "Decimals of accuracy -log10(ulp / (2|x|)) at each x.")
		.def_prop_ro("fraction_bits", [](const precision_profile& p) {
			return column(p, &precision_point::fraction_bits);
		}, own, "Fraction bits log2(|x| / ulp) at each x.")
		.def("decimals_at",
			[](const precision_profile& p, double magnitude) {
				return sw::universal::decimals_at(p, magnitude);
			}, nb::arg("magnitude"),
			"Decimals of accuracy at `magnitude`: those of the largest "
			"profiled encoding <= magnitude, and 0 outside "
			"[minpos, maxpos].")
		.def("write_csv",
			[](const precision_profile& p, const std::string& path) {
				if (!sw::universal::write_precision_csv(p, path)) {
					throw std::runtime_error(
						"PrecisionProfile.write_csv: cannot write " + path);
				}
			}, nb::arg("path"),
			"Write the profile in Universal's CSV layout, readable by "
			"universal/tools/notebooks/plot_precision_profiles.py.")
		.def("__len__", [](const precision_profile& p) {
			return p.points.size();
		})
		.def("__repr__", [](const precision_profile& p) {
			return "PrecisionProfile('" + p.label + "', " +
				std::to_string(p.points.size()) + " points, " +
				(p.exhaustive ? "exhaustive" : "sampled") + ")";
		});

	m.def("precision_profile",
		[](const std::string& type, unsigned samples_per_binade) {
			if (samples_per_binade == 0) {
				throw std::invalid_argument(
					"precision_profile: samples_per_binade must be >= 1");
			}
			for (const ProfileEntry& e : profile_table()) {
				if (type == e.key) return e.build(e.label, samples_per_binade);
			}
			throw std::invalid_argument(
				"precision_profile: unknown type '" + type +
				"' (see mpdsp.available_profile_types())");
		}, nb::arg("type"), nb::arg("samples_per_binade") = 4,
		"Profile a number system's decimals of accuracy across its range. "
		"`type` is one of `available_profile_types()`: the tutorial set "
		"(int16, Q15, fp16, lns<16,10>, bposit<16,rs,es>, and their 32-bit "
		"counterparts) plus the scalars behind every `dtype=` config. "
		"`samples_per_binade` applies only to sampled (wider than 16-bit) "
		"types.");

	m.def("available_profile_types", []() {
		std::vector<std::string> keys;
		for (const ProfileEntry& e : profile_table()) keys.emplace_back(e.key);
		return keys;
	}, "Type keys accepted by `precision_profile()`, in table order. "
	   "`half` is an alias for `fp16`.");
}
