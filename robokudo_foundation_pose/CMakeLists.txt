cmake_minimum_required(VERSION 3.0.2)
project(robokudo_foundation_pose LANGUAGES CXX)

find_package(catkin REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(CATKIN_PACKAGE_PYTHON_DESTINATION
    lib/python3/dist-packages/${PROJECT_NAME}
)
catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Build the C++ extension in its own subdirectory
add_subdirectory(src/mycpp)
install(
  TARGETS mycpp
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

