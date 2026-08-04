project(
  'pywiggle',
  'cpp',
  version: '0.1.21',
  default_options: ['cpp_std=c++17'],
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)
cpp = meson.get_compiler('cpp')

# --- pybind11 include paths (installed into the build env via pyproject.toml) --
pybind_includes = run_command(
  py, '-m', 'pybind11', '--includes',
  check: true,  # fail configure loudly if pybind11 is missing, instead of
                # passing empty flags and dying later with cryptic errors
).stdout().strip().split()

inc = include_directories('src')

sources = files(
  'src/wiggle.cpp',
  'src/wiggle_bindings.cpp', # pybind11 glue
)

# --- OpenMP -------------------------------------------------------------------
# Requirement: every binary must be OpenMP-threaded. Both branches below are
# hard-required, so configuration FAILS if OpenMP cannot be enabled -- this can
# never silently produce a serial build.
if host_machine.system() == 'darwin' and not cpp.has_argument('-fopenmp')
  # Apple clang: rejects bare -fopenmp ("clang: error: unsupported option").
  # Enable the pragmas in the frontend and link Homebrew's libomp explicitly.
  # Same recipe scikit-learn / xgboost use for their macOS wheels; in CI,
  # delocate then vendors libomp into the wheel so users don't need Homebrew.
  brew = find_program('brew', required: false)
  if not brew.found()
    error('OpenMP is required. Apple clang needs Homebrew libomp: brew install libomp')
  endif
  libomp_prefix = run_command(brew, '--prefix', 'libomp', check: true).stdout().strip()

  omp_dep = declare_dependency(
    compile_args: ['-Xpreprocessor', '-fopenmp', '-I' + libomp_prefix / 'include'],
    dependencies: cpp.find_library('omp', dirs: libomp_prefix / 'lib'),
  )
else
  # GCC / LLVM clang / MSVC: meson's builtin detection supplies the correct
  # compile AND link flags for this compiler, so no hand-rolled -fopenmp.
  omp_dep = dependency('openmp')
endif

# --- python package -----------------------------------------------------------
install_subdir(
  'pywiggle',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
)

extension = py.extension_module(
  '_wiggle',
  sources: sources,
  dependencies: [omp_dep],
  cpp_args: pybind_includes + ['-O3'],
  include_directories: inc,
  subdir: 'pywiggle',
  install: true,
)
