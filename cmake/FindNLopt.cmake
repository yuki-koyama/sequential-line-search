# Find NLopt
#
# This sets the following variables:
# NLopt_FOUND
# NLopt_INCLUDE_DIRS
# NLopt_LIBRARIES
# NLopt_DEFINITIONS
# NLopt_VERSION

find_package(PkgConfig REQUIRED)

pkg_check_modules(PC_NLopt REQUIRED NLopt)

set(NLopt_DEFINITIONS ${PC_NLopt_CFLAGS_OTHER})

find_path(NLopt_INCLUDE_DIRS
    NAMES nlopt.h
    HINTS ${PC_NLopt_INCLUDEDIR}
    PATHS "${CMAKE_INSTALL_PREFIX}/include")

find_library(NLopt_LIBRARIES
    NAMES nlopt nlopt_cxx
    HINTS ${PC_NLopt_LIBDIR})

set(NLopt_VERSION ${PC_NLopt_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NLopt
    FAIL_MESSAGE  DEFAULT_MSG
    REQUIRED_VARS NLopt_INCLUDE_DIRS NLopt_LIBRARIES
    VERSION_VAR   NLopt_VERSION)
