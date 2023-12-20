tpc_compiler = find_program('tpc-clang', dirs: ['/usr/bin/'])
tpc_objdump = find_program('tpc-llvm-objdump', dirs: ['/usr/bin/'])

tpc_test_core_library = cpp.find_library('tpc_tests_core_ext', dirs: ['/usr/lib/habanatools/'])
tpcsim_shared_library = cpp.find_library('libtpcsim_shared', dirs: ['/usr/lib/habanatools/'])
tpc_elf_reader_library = cpp.find_library('libTpcElfReader', dirs: ['/usr/lib/habanatools/'])

tests_cpp_files = files(
    'float16.cpp',
    'test_base.cpp',
    'tests_main.cpp', # TODO to remove maybe
    'gaudi_test/add_f32_test.cpp',
    'gaudi_test/avg_pool_2d_f32_test.cpp',
    'gaudi_test/batchnorm_f32_test.cpp',
    'gaudi_test/cast_gaudi_test.cpp',
    'gaudi_test/customdiv_fwd_f32_test.cpp',
    'gaudi_test/filter_fwd_2d_bf16_test.cpp',
    'gaudi_test/gather_fwd_i32_test.cpp',
    'gaudi_test/kl_div_all_test.cpp',
    'gaudi_test/leakyrelu_f32_gaudi_test.cpp',
    'gaudi_test/matrix_mul_fwd_f32_test.cpp',
    'gaudi_test/relu6_all_test.cpp',
    'gaudi_test/searchsorted_f32_test.cpp',
    'gaudi_test/sin_f32_test.cpp',
    'gaudi_test/softmax_bf16_test.cpp',
    'gaudi_test/sparse_lengths_sum_bf16_test.cpp',
    'gaudi_test/spatial_conv_f32_test.cpp',
    'gaudi2_test/avg_pool_2d_f32_gaudi2_test.cpp',
    'gaudi2_test/cast_f16_to_i16_gaudi2_test.cpp',
    'gaudi2_test/softmax_bf16_gaudi2_test.cpp',
)

tests_incdirs = include_directories(
    '../src/gaudi_src',
    '../src/gaudi2_src',
    '../src',
    'gaudi_test',
    'gaudi2_test',
    '.',
    '/usr/lib/habanatools/include',
)

tpc_kernel_tests_lib = static_library(
    'tpc_kernel_tests',
    tests_cpp_files,
    dependencies : [
        tpc_test_core_library,
        tpcsim_shared_library,
        tpc_elf_reader_library,
    ],
    link_with: [
        custom_tpc_perf_lib,
    ],
    build_by_default: true,
    include_directories : [
        tests_incdirs,
    ],
    c_args : [
        '-std=c++17', '-Wall', '-Werror', '-fno-strict-aliasing',
    ],
    cpp_args : [
        '-std=c++17', '-Wall', '-Werror', '-fno-strict-aliasing',
    ],
)
