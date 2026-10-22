
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HBM_BASE = 0xC0000000
CHANNEL_SIZE = 0x04000000
NUM_CHANNELS = 16
RESERVED_BYTES = 0x00100000


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def generate(config_path, output_dir):
    config = json.loads(config_path.read_text())

    M, N, K = (config["max_dimensions"][x] for x in ("M", "N", "K"))
    alignment = config["alignment"]
    seed = config["seed"]

    if config["dtype"] == "fp8":
        import ml_dtypes
        # Specify the exact FP8 encoding used by the kernel in benchmark.json.
        dtype = np.dtype(getattr(ml_dtypes, "float8_" + config["fp8_format"]))
    else:
        dtype = np.dtype("<f2")

    shapes = {
        "A": (M, K),
        "B": (K, N),
        "C": (M, N),
    }

    matrices = {}
    offset = RESERVED_BYTES

    for name, shape in shapes.items():
        offset = align_up(offset, alignment)
        size = shape[0] * shape[1] * dtype.itemsize
        matrices[name] = {"offset": offset, "size_bytes": size}
        offset += size

    if offset > CHANNEL_SIZE:
        raise ValueError("Matrix data exceeds HBM channel capacity")

    output_dir.mkdir(parents=True, exist_ok=True)

    a_start = matrices["A"]["offset"]
    b_end = matrices["B"]["offset"] + matrices["B"]["size_bytes"]

    phdrs = []
    sections = []

    for cid in range(NUM_CHANNELS):
        rng = np.random.default_rng(seed + cid)

        image_path = output_dir / f"channel_{cid:02d}.bin"
        payload_path = output_dir / f"matrix_{cid:02d}.bin"
        object_path = output_dir / f"matrix_{cid:02d}.o"

        with open(image_path, "wb") as f:
            f.truncate(offset)

            for name in ("A", "B"):
                values = rng.uniform(
                    -100, 100, size=shapes[name]
                ).astype(dtype)

                f.seek(matrices[name]["offset"])
                f.write(values.tobytes())

        # Extract only A and B, not the reserved prefix or C.
        with open(image_path, "rb") as f:
            f.seek(a_start)
            payload_path.write_bytes(f.read(b_end - a_start))

        section = f".matrix_payload_{cid:02d}"
        phdr = f"matrix_{cid:02d}"
        address = HBM_BASE + cid * CHANNEL_SIZE + a_start

        subprocess.run([
            "riscv32-unknown-elf-objcopy",
            "-I", "binary",
            "-O", "elf32-littleriscv",
            "-B", "riscv",
            "--rename-section",
            f".data={section},alloc,load,readonly,data,contents",
            str(payload_path),
            str(object_path),
        ], check=True)

        phdrs.append(f"  {phdr} PT_LOAD;")
        sections.append(
            f"  {section} 0x{address:08X} : "
            f"{{ KEEP(*({section})) }} :{phdr}"
        )

    (output_dir / "hbm_phdrs.ld").write_text(
        "\n".join(phdrs) + "\n"
    )

    (output_dir / "hbm_sections.ld").write_text(
        "\n".join(sections) + "\n"
    )

    layout = {
        "dimensions": {"M": M, "N": N, "K": K},
        "dtype": config["dtype"],
        "hbm_base": HBM_BASE,
        "channel_size": CHANNEL_SIZE,
        "num_channels": NUM_CHANNELS,
        "matrices": matrices,
    }

    (output_dir / "layout.json").write_text(
        json.dumps(layout, indent=2) + "\n"
    )

    header = f"""\
#pragma once
#include <cstddef>
#include <cstdint>

namespace benchmark_layout {{

constexpr size_t M = {M};
constexpr size_t N = {N};
constexpr size_t K = {K};

constexpr uintptr_t HBM_BASE = 0x{HBM_BASE:08X};
constexpr uintptr_t CHANNEL_SIZE = 0x{CHANNEL_SIZE:08X};

constexpr size_t A_OFFSET = 0x{matrices["A"]["offset"]:X};
constexpr size_t B_OFFSET = 0x{matrices["B"]["offset"]:X};
constexpr size_t C_OFFSET = 0x{matrices["C"]["offset"]:X};

template<typename T>
T* matrix_a(uint32_t cid) {{
    return reinterpret_cast<T*>(HBM_BASE + cid * CHANNEL_SIZE + A_OFFSET);
}}

template<typename T>
T* matrix_b(uint32_t cid) {{
    return reinterpret_cast<T*>(HBM_BASE + cid * CHANNEL_SIZE + B_OFFSET);
}}

template<typename T>
T* matrix_c(uint32_t cid) {{
    return reinterpret_cast<T*>(HBM_BASE + cid * CHANNEL_SIZE + C_OFFSET);
}}

}} // namespace benchmark_layout
"""

    (output_dir / "benchmark_layout.hpp").write_text(header)

    print("Generated HBM images, objects, linker fragments and header")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generate(args.config, args.output)
