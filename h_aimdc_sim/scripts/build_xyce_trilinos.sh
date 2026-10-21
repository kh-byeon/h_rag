#!/usr/bin/env bash
# Build Trilinos (MPI) then Xyce 7.8 from official tarballs (no apt — install deps in Dockerfile).
set -euo pipefail

TRILINOS_VER="${TRILINOS_VER:-trilinos-release-15-0-0}"
XYCE_TAG="${XYCE_TAG:-Release-7.8.0}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p /opt/src
cd /opt/src

# --- Trilinos ---
if [[ ! -d "Trilinos-${TRILINOS_VER}" ]]; then
  wget -q "https://github.com/trilinos/Trilinos/archive/refs/tags/${TRILINOS_VER}.tar.gz" -O trilinos.tgz
  tar -xf trilinos.tgz
fi
cd "Trilinos-${TRILINOS_VER}"
cmake -S . -B ../trilinos-build -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/opt/trilinos \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_CXX_STANDARD=17 \
  -D TPL_ENABLE_MPI=ON \
  -D MPI_BASE_DIR=/usr \
  -D TPL_ENABLE_BLAS=ON \
  -D TPL_ENABLE_LAPACK=ON \
  -D TPL_BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -D TPL_LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
  -D TPL_ENABLE_AMD=ON \
  -D TPL_ENABLE_UMFPACK=ON \
  -D UMFPACK_INCLUDE_DIRS=/usr/include/suitesparse \
  -D AMD_INCLUDE_DIRS=/usr/include/suitesparse \
  -D TPL_ENABLE_FFTW=ON \
  -D FFTW_INCLUDE_DIRS=/usr/include \
  -D FFTW_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu \
  -D Trilinos_ENABLE_Fortran=ON \
  -D Trilinos_ENABLE_OpenMP=ON \
  -D Trilinos_ENABLE_TESTS=OFF \
  -D Trilinos_ENABLE_EXAMPLES=OFF \
  -D Trilinos_ENABLE_COMPLEX=ON \
  -D Trilinos_ENABLE_Epetra=ON \
  -D Trilinos_ENABLE_EpetraExt=ON \
  -D EpetraExt_BUILD_BTF=ON \
  -D EpetraExt_BUILD_EXPERIMENTAL=ON \
  -D EpetraExt_BUILD_GRAPH_REORDERINGS=ON \
  -D Trilinos_ENABLE_Teuchos=ON \
  -D Trilinos_ENABLE_Sacado=ON \
  -D Trilinos_ENABLE_Amesos=ON \
  -D Trilinos_ENABLE_AztecOO=ON \
  -D Trilinos_ENABLE_Ifpack=ON \
  -D Trilinos_ENABLE_NOX=ON \
  -D NOX_ENABLE_Stratimikos=OFF \
  -D Trilinos_ENABLE_Belos=ON \
  -D Trilinos_ENABLE_TrilinosCouplings=ON \
  -D Trilinos_ENABLE_MueLu=OFF \
  -D Stratimikos_ENABLE_MueLu=OFF \
  -D Trilinos_ENABLE_Teko=OFF \
  -D Trilinos_ENABLE_Triutils=ON \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=OFF
cmake --build ../trilinos-build -j"${JOBS}"
cmake --install ../trilinos-build
cd /opt/src

# --- Xyce 7.8 ---
if [[ ! -d "Xyce-${XYCE_TAG}" ]]; then
  wget -q "https://github.com/Xyce/Xyce/archive/refs/tags/${XYCE_TAG}.tar.gz" -O xyce.tgz
  tar -xf xyce.tgz
fi
cd "Xyce-${XYCE_TAG}"
cmake -S . -B ../xyce-build -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_Fortran_COMPILER=mpif90 \
  -D FLEX_INCLUDE_DIR=/usr/include \
  -D Xyce_ENABLE_MPI=ON \
  -D Trilinos_DIR=/opt/trilinos/lib/cmake/Trilinos \
  -D Xyce_PLUGIN_SUPPORT=ON \
  -D Xyce_USE_SYSTEM_FFTW=ON
cmake --build ../xyce-build -j"${JOBS}"
cmake --install ../xyce-build

rm -rf /opt/src /opt/trilinos-build /opt/xyce-build
