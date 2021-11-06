project(ullc_diff_drive)
cmake_minimum_required(VERSION 3.5)

find_package(catkin_simple REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

cs_install_scripts(
  src/sim.py
  src/cli_control.py
)

cs_install()
cs_export()
