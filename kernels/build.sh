#!/bin/bash
set -x

mkdir -p "${1}/src"
pushd "${1}/src"

cmake "${2}/../kernels"
make

popd
