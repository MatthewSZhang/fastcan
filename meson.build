project(
  'fastcan',
  'c', 'cython',
  version: '0.4.0',
  license: 'MIT',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=release',
  ],
)

py = import('python').find_installation(pure: false)

install_subdir('fastcan', install_dir: py.get_install_dir())

subdir('fastcan')
