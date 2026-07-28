#!/usr/bin/env bash
set -euo pipefail

# Rebuild the one native dependency on gu30.  The upstream release wheel was
# built against GLIBC_2.32, while gu30 provides glibc 2.28.  This build is
# deliberately restricted to the deployed RTX 4090 architecture (SM 8.9).

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="/home/caoxiangyu/.cache/janus-ts"
source_commit="4f6ae4e26ae5fe8af9372f8d312ab25cc4595223"
cuda_env="/home/caoxiangyu/.cache/micromamba/envs/janus-ts-cuda128"
micromamba="/home/caoxiangyu/.local/bin/micromamba"
uv_bin="/home/caoxiangyu/.local/bin/uv"
wheel_dir="${cache_root}/wheels"

mkdir -p "${cache_root}/build" "${wheel_dir}"

if [[ ! -x "${micromamba}" ]]; then
  echo "missing micromamba: ${micromamba}" >&2
  exit 1
fi
if [[ ! -x "${uv_bin}" || ! -x "${project_root}/.venv/bin/python" ]]; then
  echo "run uv sync in ${project_root} before building" >&2
  exit 1
fi

if [[ ! -d "${cuda_env}/conda-meta" ]]; then
  "${micromamba}" create -y -p "${cuda_env}" \
    -c nvidia -c conda-forge \
    cuda-nvcc=12.8.93 python=3.11.15 clangxx_linux-64=17.0.6
elif [[ ! -x "${cuda_env}/bin/nvcc" \
     || ! -x "${cuda_env}/bin/clang++" \
     || ! -f "${cuda_env}/include/python3.11/Python.h" ]]; then
  "${micromamba}" install -y -p "${cuda_env}" \
    -c nvidia -c conda-forge \
    cuda-nvcc=12.8.93 python=3.11.15 clangxx_linux-64=17.0.6
fi

build_root="$(mktemp -d "${cache_root}/build/causal-conv1d.XXXXXXXX")"
source_root="${build_root}/source"
git clone --quiet https://github.com/Dao-AILab/causal-conv1d.git "${source_root}"
git -C "${source_root}" checkout --quiet "${source_commit}"
git -C "${source_root}" apply \
  "${project_root}/patches/causal-conv1d-v1.6.2.post1-sm89.patch"

export CUDA_HOME="${cuda_env}"
export CC=/usr/bin/gcc
export CXX="${cuda_env}/bin/clang++ --gcc-toolchain=/usr --sysroot=/ \
-I${cuda_env}/include/python3.11 \
-I${cuda_env}/targets/x86_64-linux/include"
export CUDAHOSTCXX=/usr/bin/g++
export MAX_JOBS=1
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE
export CAUSAL_CONV1D_LOCAL_VERSION=cu128torch2.9cxx11abiTRUEglibc228

cd "${project_root}"
"${uv_bin}" build --wheel --no-build-isolation \
  --out-dir "${wheel_dir}" "${source_root}"

wheel_path="$(find "${wheel_dir}" -maxdepth 1 -type f \
  -name 'causal_conv1d-1.6.2.post1+cu128torch2.9cxx11abitrueglibc228-*.whl' \
  -print -quit)"
if [[ -z "${wheel_path}" ]]; then
  echo "expected wheel was not produced in ${wheel_dir}" >&2
  exit 1
fi

sha256sum "${wheel_path}"
echo "source_commit=${source_commit}"
echo "build_root=${build_root}"
