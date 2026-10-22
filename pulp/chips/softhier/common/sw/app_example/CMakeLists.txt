# Specify the source files for the "src" module
set(SRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# Make sources and include paths visible to the main CMakeLists.txt
set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
