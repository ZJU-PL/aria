cmake_minimum_required(VERSION 3.10)
project(scsat_app LANGUAGES CXX)

# Find Z3 - try CONFIG mode first (for Homebrew installations with CMake config)
# Homebrew typically installs CMake config files in /opt/homebrew/lib/cmake/z3
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(Z3 CONFIG)

# Check if targets were created
if(TARGET z3::libz3)
    set(_Z3_TARGET z3::libz3)
elseif(TARGET Z3::libz3)
    set(_Z3_TARGET Z3::libz3)
elseif(TARGET Z3::z3)
    set(_Z3_TARGET Z3::z3)
elseif(TARGET z3::z3)
    set(_Z3_TARGET z3::z3)
else()
    # Fallback: use find_library approach similar to reference project
    find_library(Z3_LIBRARIES NAMES z3 HINTS /opt/homebrew/lib)
    find_path(Z3_INCLUDES NAMES z3++.h HINTS /opt/homebrew/include)
    if(Z3_LIBRARIES AND Z3_INCLUDES)
        message(STATUS "Found Z3 via find_library: ${Z3_LIBRARIES}")
        include_directories(${Z3_INCLUDES})
        add_library(z3_lib SHARED IMPORTED)
        set_target_properties(z3_lib PROPERTIES
            IMPORTED_LOCATION ${Z3_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${Z3_INCLUDES}
        )
        set(_Z3_TARGET z3_lib)
    else()
        message(FATAL_ERROR "Unable to find Z3 target or library (expected Z3::libz3, Z3::z3, or z3 library).")
    endif()
endif()

add_library(scsat_app STATIC
    unary_check.cpp
    disjunctive_check.cpp
    conjunctive_check.cpp
)
target_include_directories(scsat_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scsat_app PUBLIC ${_Z3_TARGET})
target_compile_features(scsat_app PUBLIC cxx_std_17)

add_executable(scsat_example example.cpp)
target_link_libraries(scsat_example PRIVATE scsat_app ${_Z3_TARGET})
