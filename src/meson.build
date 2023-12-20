
src_cpp_files = files(
    'entry_points.cpp',
    'spatial_reduction_kernels.cpp',
    'gaudi_src/add_f32.cpp',
    'gaudi_src/avg_pool_2d_f32.cpp',
    'gaudi_src/batch_norm_f32.cpp',
    'gaudi_src/cast_gaudi.cpp',
    'gaudi_src/customdiv_fwd_f32.cpp',
    'gaudi_src/filter_fwd_2d_bf16.cpp',
    'gaudi_src/gather_fwd_i32.cpp',
    'gaudi_src/kl_div_all.cpp',
    'gaudi_src/leakyrelu_f32_gaudi.cpp',
    'gaudi_src/matrix_mul_fwd_f32.cpp',
    'gaudi_src/printf_test.cpp',
    'gaudi_src/relu6_all.cpp',
    'gaudi_src/searchsorted_f32.cpp',
    'gaudi_src/sin_f32.cpp',
    'gaudi_src/softmax_bf16.cpp',
    'gaudi_src/sparse_lengths_sum_bf16.cpp',
    'gaudi_src/spatial_conv_f32.cpp',
    'gaudi2_src/avg_pool_2d_f32_gaudi2.cpp',
    'gaudi2_src/cast_f16_to_i16_gaudi2.cpp',
    'gaudi2_src/softmax_bf16_gaudi2.cpp',
)

tests_incdirs = include_directories(
    'gaudi_src',
    'gaudi2_src',
    '.',
    '/usr/lib/habanatools/include',
)

run_command('../kernels/build.sh', meson.project_build_root(), check: true)

kernels = [
    cpp.find_library('add_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('avg_pool_2d_bwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('avg_pool_2d_bwd_f32_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('avg_pool_2d_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('avg_pool_2d_fwd_f32_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('batch_norm_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('cast_bf16_to_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('cast_f16_to_i16_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('cast_f32_to_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('customdiv_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('filter_fwd_2d_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('gather_fwd_dim0_i32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('gather_fwd_dim1_i32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('kl_div_bwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('kl_div_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('kl_div_fwd_f32_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('leakyrelu_f32_gaudi', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('matrix_mul_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('printf_test', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu6_bwd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu6_bwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu6_fwd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu6_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu_bwd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu_bwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu_fwd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('relu_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('searchsorted_fwd_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('sin_f32', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('softmax_fcd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('softmax_fcd_bf16_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('softmax_non_fcd_bf16', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('softmax_non_fcd_bf16_gaudi2', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('sparse_lengths_sum_bf16_2D_f32_embed', dirs: [meson.project_build_root() + '/src']),
    cpp.find_library('spatial_conv_f32', dirs: [meson.project_build_root() + '/src']),
]

custom_tpc_perf_lib = shared_library(
    'custom_tpc_perf_lib',
    src_cpp_files,
    dependencies : [
        tpc_test_core_library,
        tpcsim_shared_library,
        tpc_elf_reader_library,
        kernels,
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
