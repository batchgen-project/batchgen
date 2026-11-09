#!/usr/bin/env bash
# Usage: CXX=<repo>/scripts/cache_cxx.sh with CACHE_CXX_LAUNCHER=<ccache|sccache>
#        (and optionally CACHE_CXX_COMPILER, default g++) in the environment.
#
# A plain-executable C++ compiler that runs the real compiler under a compile
# cache. torch >= 2.15 shell-quotes CXX as ONE program and prepends ccache
# itself, so a multi-word CXX="ccache g++" becomes `ccache 'ccache g++'`;
# torch 2.9 adds no launcher at all. Keeping the launcher inside this
# executable (and setting TORCH_NO_COMPILER_WRAPPER so newer torch does not
# add a second one) gives every torch version the same single-launcher line.
launcher="${CACHE_CXX_LAUNCHER:?CACHE_CXX_LAUNCHER must name the compile cache (ccache or sccache) when CXX is cache_cxx.sh}"
exec "$launcher" "${CACHE_CXX_COMPILER:-g++}" "$@"
