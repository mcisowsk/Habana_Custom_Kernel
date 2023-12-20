cmake_minimum_required(VERSION 3.0.2)
project(tpc_kernels)

set(TPC_COMPILER_PATH "/usr/bin/tpc-clang" CACHE STRING "Path to TPC C/C++ compiler")
set(TPC_DISASSEMBLER_PATH "/usr/bin/tpc-llvm-objdump" CACHE STRING "Path to TPC disassembler")
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "Path to TPC includes")
set(TPC_CURRENT_LIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "Path to current includes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Werror -fno-strict-aliasing")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-ggdb -O2")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build type selected for Debug")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()


# Add TPC-C source files to be compiled by TPC Gaudi compiler and
# linked into tpc_kernels executable.
file(GLOB TPC_C_GAUDI_SOURCES_TO_EMBED gaudi/*.c)

# Add TPC-C source files to be compiled by TPC Gaudi2 compiler and
# linked into tpc_kernels executable.
file(GLOB TPC_C_GAUDI2_SOURCES_TO_EMBED gaudi2/*.c)

# Add TPC assembly files to be compiled by TPC Goya assembler and
# linked into tpc_kernels executable.
file(GLOB TPC_ASSEMBLY_TO_EMBED  *.tpcasm)


# This routine does the following:
# 1. Copies the TPC-C source file to 'bin' for reference.
# 2. Compiles sources using TPC-C compiler (two times - one is for testing of compiler regression).
# 3. Wraps compiler output in X86 elf.
# 4. Links the X86 elf into the executable
# 5. Disassemble the file for debug purposes
function(CompileTPCCWithClang TPC_C_SOURCES_TO_EMBED TPC_HEADERS deviceName)
    foreach(SOURCE_FILE  ${TPC_C_SOURCES_TO_EMBED})
        get_filename_component(SOURCE_NAME_NO_PATH_NO_SUFFIX ${SOURCE_FILE} NAME_WE)
        get_filename_component(SOURCE_FILE_NAME_NO_PATH ${SOURCE_FILE} NAME)
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND "${SOURCE_FILE_NAME_NO_PATH}" STREQUAL "filter_2d_i8_w33_s11.c")
            set (OPT_FLAG "-O1")
        elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set (OPT_FLAG "-O0")
        endif()
        add_custom_command(OUTPUT "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
            DEPENDS ${SOURCE_FILE} ${TPC_COMPILER_PATH} ${TPC_HEADERS}
                COMMAND cp ${SOURCE_FILE}  ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=${deviceName} ${OPT_FLAG} ${DEBUG_FLAG} -I${TPC_INCLUDE_DIR}/ -I${TPC_CURRENT_LIST_DIR} ./${SOURCE_FILE_NAME_NO_PATH} -c
                COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
                COMMAND  ${TPC_DISASSEMBLER_PATH} --triple tpc -d -j .text --no-show-raw-insn --no-leading-addr --mcpu=${deviceName} ./${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o > ${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.tpcasm)
        add_library(${SOURCE_NAME_NO_PATH_NO_SUFFIX} STATIC "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o")
        set_target_properties( ${SOURCE_NAME_NO_PATH_NO_SUFFIX} PROPERTIES LINKER_LANGUAGE C  )
    endforeach()
endfunction(CompileTPCCWithClang)

CompileTPCCWithClang("${TPC_C_GAUDI_SOURCES_TO_EMBED}" "${TPC_GAUDI_HEADERS}" gaudi)
CompileTPCCWithClang("${TPC_C_GAUDI2_SOURCES_TO_EMBED}" "${TPC_GAUDI2_HEADERS}" gaudi2)
