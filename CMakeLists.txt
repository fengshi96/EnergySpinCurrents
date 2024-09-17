cmake_minimum_required(VERSION 3.20)
project(LanczosCond)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-fPIC -Wno-gcc-compat)



# Flags of sources, headers and libraries
include_directories(
        # /Users/shifeng/Public/hdf5-1.8.23/include
        /usr/local/include/eigen3
        /opt/homebrew/opt/libomp/include
        ./src
)

link_directories(
        # /Users/shifeng/Public/hdf5-1.8.23/lib
        /usr/local/include/eigen3 
        /opt/homebrew/opt/libomp/lib
        ./src
)

# set(Libs -llapack -lblas -lhdf5_hl -lhdf5 -lomp)
set(Libs -llapack -lblas -lomp)
set(SOURCE_FILES main.cpp)

add_executable(LanczosCond main.cpp)
target_link_libraries(LanczosCond ${Libs} "-framework Accelerate")
