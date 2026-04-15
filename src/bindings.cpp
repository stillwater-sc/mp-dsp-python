// bindings.cpp: nanobind module definition for mpdsp._core
//
// This is the top-level entry point for the C++ extension module.
// Sub-modules are registered by bind_*() functions in separate files.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations for sub-module binders
void bind_signals(nb::module_& m);
void bind_quantization(nb::module_& m);
void bind_spectral(nb::module_& m);

NB_MODULE(_core, m) {
	m.doc() = "mpdsp C++ core: mixed-precision DSP bindings via nanobind";

	bind_signals(m);
	bind_quantization(m);
	bind_spectral(m);
}
