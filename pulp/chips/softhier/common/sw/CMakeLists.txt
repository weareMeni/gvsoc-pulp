cmake_minimum_required(VERSION 3.13)

# Project name and language
project(SoftHier LANGUAGES C CXX)

# Specify the RISC-V compiler
set(COMPILER "riscv32-unknown-elf-g++")
set(OBJDUMPER "riscv32-unknown-elf-objdump")

# Default SRC_DIR to "app_example" if not provided
set(SRC_DIR ${SRC_DIR} CACHE STRING "Source directory")
if (NOT SRC_DIR)
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/app_example")
endif()

# AUTO-DETECT: Find all .c files in the target folder automatically
file(GLOB SOURCES CONFIGURE_DEPENDS "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp")


if (NOT INCLUDE_DIRS)
    set(INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/runtime/include)
endif()

# Set the linker script
set(LINKER_SCRIPT ${CMAKE_SOURCE_DIR}/runtime/flex_memory.ld)


add_custom_command(
    OUTPUT softhier.elf
    COMMAND ${COMPILER} -I ${CMAKE_SOURCE_DIR}/runtime/include -I ${INCLUDE_DIRS} -I ${CMAKE_BINARY_DIR}/hbm -T ${LINKER_SCRIPT} -nostartfiles -mabi=ilp32d -mcmodel=medlow -march=${RISCV_ARCH} -g -O3 -ffunction-sections -ffast-math -fno-builtin -fno-tree-vectorize -fno-common -ffunction-sections -Wl,--gc-sections -o softhier.elf ${CMAKE_SOURCE_DIR}/runtime/flex_start.s ${SOURCES}
    COMMENT "Generating softhier.elf"
)

add_custom_target(build_output ALL DEPENDS softhier.elf)

add_custom_command(
    OUTPUT softhier.dump
    DEPENDS softhier.elf
    COMMAND ${OBJDUMPER} -d softhier.elf > softhier.dump
    COMMENT "Creating disassembly output.dump"
)

add_custom_target(dump ALL DEPENDS softhier.dump)


