// bindings.cpp: nanobind module definition for mpdsp._core
//
// This is the top-level entry point for the C++ extension module.
// Sub-modules are registered by bind_*() functions in separate files.

#include <nanobind/nanobind.h>

#include <sw/dsp/version.hpp>

namespace nb = nanobind;

// Forward declarations for sub-module binders
void bind_signals(nb::module_& m);
void bind_quantization(nb::module_& m);
void bind_spectral(nb::module_& m);
void bind_filters(nb::module_& m);
void bind_conditioning(nb::module_& m);
void bind_estimation(nb::module_& m);
void bind_image(nb::module_& m);
void bind_types(nb::module_& m);

NB_MODULE(_core, m) {
	m.doc() = "mpdsp C++ core: mixed-precision DSP bindings via nanobind";

	// Expose the upstream C++ library version the wheel was built against.
	// Python mirror is mpdsp.__dsp_version__. This is the runtime-checkable
	// analogue to the build-time lockstep between this package's __version__
	// and the mixed-precision-dsp release it wraps.
	m.attr("dsp_version") = sw::dsp::version_string;
	m.attr("dsp_version_info") = nb::make_tuple(
		sw::dsp::version_major,
		sw::dsp::version_minor,
		sw::dsp::version_patch);

	bind_signals(m);
	bind_quantization(m);
	bind_spectral(m);
	bind_filters(m);
	bind_conditioning(m);
	bind_estimation(m);
	bind_image(m);
	bind_types(m);
}
