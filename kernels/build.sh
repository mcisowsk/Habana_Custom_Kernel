#!/bin/bash
set -x

pushd "${1}/src"

cmake ../../kernels
make

popd