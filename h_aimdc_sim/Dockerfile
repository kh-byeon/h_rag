# H-AIMDC accelerator simulation image: Ubuntu 22.04 + Trilinos + Xyce 7.8 (MPI) + OpenVAF
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Runtime + build dependencies: Kakao mirror + apt retries + dynamic libssl sync
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    for attempt in 1 2 3; do \
        apt-get update --fix-missing && break; \
        [ "$attempt" -eq 3 ] && exit 1; \
        sleep 15; \
    done && \
    SSL_VER=$(apt-cache show libssl-dev | grep '^Version:' | awk '{print $2}' | head -n 1) && \
    apt-get install -y --allow-downgrades libssl3=${SSL_VER} libssl-dev=${SSL_VER} && \
    apt-get install -y --no-install-recommends \
        build-essential ninja-build git wget ca-certificates curl \
        python3 gfortran flex bison openmpi-bin libopenmpi-dev \
        libblas-dev liblapack-dev libopenblas-dev \
        libfftw3-dev libsuitesparse-dev \
        zlib1g-dev libreadline-dev libnetcdf-dev libfl-dev \
        autoconf automake libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*

# CMake 3.23+ required by Trilinos / Xyce (Ubuntu 22.04 apt ships 3.22.x)
RUN cd /tmp \
    && wget -q https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh \
    && chmod +x cmake-3.28.3-linux-x86_64.sh \
    && ./cmake-3.28.3-linux-x86_64.sh --prefix=/usr/local --skip-license \
    && rm -f cmake-3.28.3-linux-x86_64.sh

# Verilog-AMS standard includes (for OpenVAF / VA models)
RUN mkdir -p /opt/va-std \
    && curl -fsSL https://raw.githubusercontent.com/steveicarus/iverilog/master/disciplines.vams \
        -o /opt/va-std/disciplines.vams \
    && curl -fsSL https://raw.githubusercontent.com/steveicarus/iverilog/master/constants.vams \
        -o /opt/va-std/constants.vams

# OpenVAF prebuilt binary (Linux x86-64)
RUN cd /tmp \
    && wget -q https://openva.fra1.cdn.digitaloceanspaces.com/openvaf_23_5_0_linux_amd64.tar.gz \
    && tar -xzf openvaf_23_5_0_linux_amd64.tar.gz \
    && OVF="$(find /tmp -maxdepth 3 -type f -name openvaf 2>/dev/null | head -1)" \
    && test -n "${OVF}" \
    && install -m0755 "${OVF}" /usr/local/bin/openvaf \
    && rm -f /tmp/openvaf_23_5_0_linux_amd64.tar.gz

ENV OPENVAF_HOME=/usr/local \
    VAMS_INC=/opt/va-std \
    PATH=/usr/local/bin:${PATH}

# Trilinos 15 + Xyce 7.8 from source (stable reproducible path vs. ad-hoc binaries)
COPY scripts/build_xyce_trilinos.sh /tmp/build_xyce_trilinos.sh
RUN chmod +x /tmp/build_xyce_trilinos.sh && /tmp/build_xyce_trilinos.sh && rm -f /tmp/build_xyce_trilinos.sh

# Simulation driver + example netlists (copy to /work at runtime or bind-mount your files over /work)
WORKDIR /work
COPY run_sim.sh /usr/local/bin/run_sim.sh
RUN chmod +x /usr/local/bin/run_sim.sh
COPY examples/ /opt/h_aimdc_sim/examples/

# Optional: default CMD shows help
CMD ["/bin/bash", "-lc", "echo 'H-AIMDC sim: mount netlists under /work, then: run_sim.sh'; exec bash"]