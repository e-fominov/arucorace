cmake_minimum_required(VERSION 3.5)

project(arucorace)

if (MSVC)
    set(CMAKE_CXX_STANDARD 14)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED on)

#message("output dir: ${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCV)
find_package(Threads REQUIRED)
if (MSVC)
    set(OpenCV_STATIC ON)
endif()
find_package(OpenCV 4 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/3rdparty/include
    ${OpenCV_INCLUDE_DIRS}
    )

add_subdirectory(src)
