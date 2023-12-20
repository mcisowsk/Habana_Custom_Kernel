project(
    'tpc_kernels_db',
    ['c', 'cpp'],
    default_options : [
        'b_ndebug=if-release',
        'c_std=gnu17',
        'cpp_std=gnu++20',
    ],
    meson_version : '>=0.57',
)

env = environment()
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

subdir('src')
subdir('tests')
