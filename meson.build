project('pywiggle',
        ['cpp'],
        default_options : ['cpp_std=c++17'])

py = import('python').find_installation(pure: false)


# Get pybind11 include paths dynamically
pybind_includes = run_command(py, '-m', 'pybind11', '--includes').stdout().strip().split()


inc    = include_directories('src')

sources = files(
  'src/wiggle.cpp',          
  'src/wiggle_bindings.cpp', # pybind11 glue
)

# --- find OpenMP ----------------
omp_dep = dependency('openmp')

# Enable OpenMP for C++ manually
add_project_arguments('-fopenmp', language: 'cpp')
add_project_link_arguments('-fopenmp', language: 'cpp')

install_subdir('pywiggle', install_dir : py.get_install_dir())

extension = py.extension_module(
    '_wiggle',
    sources: sources,
    dependencies: [omp_dep],
    cpp_args: pybind_includes + ['-O3', '-fopenmp'],
    include_directories: inc,
    subdir: 'pywiggle',   
    install: true,
)

