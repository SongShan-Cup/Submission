-- Same direct package requirements as the supplied InfiniCore/InfiniLM baseline.
-- Run on an online Linux preparation machine with a compatible host toolchain.
add_rules("mode.debug", "mode.release")
add_requires("boost", {configs = {stacktrace = true}})
add_requires("pybind11")
add_requires("openmp")
target("h3_package_probe")
    set_kind("binary")
    set_languages("cxx17")
    add_files("main.cpp")
    add_packages("boost", "pybind11", "openmp")
