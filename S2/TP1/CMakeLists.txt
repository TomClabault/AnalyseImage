cmake_minimum_required(VERSION 3.16s)
project(TP1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "%{Qt:QT_INSTALL_PREFIX};E:\\Programmes\\opencv-4.6.0\\build")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native -pthread")
    set(PLATFORM_SPECIFIC_LIBS "-lpthread")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

include_directories(inc)

set(SRCS
    main.cpp
    filters.cpp
	filters.hpp
	settings.cpp
	settings.hpp
)


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main ${SRCS})

target_link_libraries(main ${OpenCV_LIBS} )
