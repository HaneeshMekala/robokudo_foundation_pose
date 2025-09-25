cmake_minimum_required(VERSION 3.15)
project(mycpp LANGUAGES CXX)


# ────────────────────────────  Build options  ────────────────────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)            # use -std=c++14, not gnu++14
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # needed for Python extension

# ─────────────────────────────  Dependencies  ────────────────────────────
find_package(Boost   REQUIRED COMPONENTS system program_options)
find_package(OpenMP  REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(pybind11 REQUIRED)
set(CATKIN_PACKAGE_LIB_DESTINATION lib/${PROJECT_NAME})


# If we’re inside a catkin build the vars are already defined
if(catkin_FOUND)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

# ──────────────────────────────  Sources  ────────────────────────────────
# All .cpp files in src/, **except** the pybind front‑end (added separately).
file(GLOB MY_SRC CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/src/*.cpp)

list(REMOVE_ITEM MY_SRC ${PROJECT_SOURCE_DIR}/src/app/pybind_api.cpp)

# ────────────────────────────  Pybind11 module  ──────────────────────────
pybind11_add_module(mycpp
    src/app/pybind_api.cpp        # exposes the API
    ${MY_SRC}                     # core C++ implementation files
)

target_include_directories(mycpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    Eigen3::Eigen
)

target_link_libraries(mycpp PRIVATE
    Boost::boost
    Boost::system
    Boost::program_options
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${catkin_LIBRARIES}           # harmless if catkin isn’t present
)
# ─────────────── 2) Copy into devel/lib/<package> on every build ───────────────
# after your catkin_package(…) and pybind11_add_module(…)
set(CATKIN_PACKAGE_LIB_DESTINATION lib/${PROJECT_NAME})  # PROJECT_NAME here is foundationpose
add_custom_command(TARGET mycpp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}

  COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:mycpp>
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/$<TARGET_FILE_NAME:mycpp>

  COMMENT "Copying mycpp.so into devel/${CATKIN_PACKAGE_PYTHON_DESTINATION}"
)


# ─────────────────────────────  Installation  ────────────────────────────
# If we’re inside catkin, CATKIN_PACKAGE_PYTHON_DESTINATION is already set;
# otherwise fall back to the normal site‑packages path.
if(NOT DEFINED CATKIN_PACKAGE_PYTHON_DESTINATION)
  set(CATKIN_PACKAGE_PYTHON_DESTINATION
      lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
endif()
message(STATUS "CATKIN_PACKAGE_PYTHON_DESTINATION = ${CATKIN_PACKAGE_PYTHON_DESTINATION}")
install(TARGETS mycpp
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION})


