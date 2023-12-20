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

tpc_compiler = find_program('tpc-clang', dirs: ['/usr/bin/'])
tpc_objdump = find_program('tpc-llvm-objdump', dirs: ['/usr/bin/'])

tpc_test_core_library = cpp.find_library('tpc_tests_core_ext', dirs: ['/usr/lib/habanatools/'])
tpcsim_shared_library = cpp.find_library('libtpcsim_shared', dirs: ['/usr/lib/habanatools/'])
tpc_elf_reader_library = cpp.find_library('libTpcElfReader', dirs: ['/usr/lib/habanatools/'])

subdir('src')
subdir('tests')
