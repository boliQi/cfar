cmake_minimum_required(VERSION 2.8)

project(cfar)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -fopenmp -fno-strict-aliasing -O2 -g -fPIC -Wall -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fno-strict-aliasing -O2 -g -fPIC -Wall -Wno-unused-but-set-variable")
message(STATUS "Use ${CMAKE_CXX_COMPILER}${CMAKE_CXX_FLAGS}  as CMAKE_CXX_FLAGS")

#头文件
include_directories(include  /usr/local/clblas/include)

#源码
file(GLOB_RECURSE SOURCES "src/*" )

#link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")
#set(CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib" ${CMAKE_LIBRARY_PATH})


#可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}  
	m  
	pthread	
    OpenCL
    clBLAS
    clFFT
    )
