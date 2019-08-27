#include "core.hpp"

#ifdef SEQUENTIAL_LINE_SEARCH_PHOTO_DIM_SUBSET
#define PHOTO_DIM 2
#else
#define PHOTO_DIM 6
#endif

Core::Core() : dim(PHOTO_DIM) {}
