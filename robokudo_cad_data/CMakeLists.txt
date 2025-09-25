cmake_minimum_required(VERSION 3.0.2)
project(robokudo_cad_data)

find_package(catkin REQUIRED)
catkin_package()  # no libraries or executables

install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
  PATTERN ".svn" EXCLUDE)

