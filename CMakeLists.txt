cmake_minimum_required ( VERSION 3.5 )

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;")
if (NOT CMAKE_BUILD_TYPE)
    message("Defaulting to release build.")
    set(CMAKE_BUILD_TYPE Release)
endif()

project( gpgpu )

add_executable( gpgpu_test
                ./main.cpp )

set_property( TARGET
                     gpgpu_test
            PROPERTY
                     CXX_STANDARD 17 )

add_definitions ( -DSRC_ROOT=${CMAKE_SOURCE_DIR} )

target_compile_options(gpgpu_test PUBLIC "$<$<CONFIG:DEBUG>:-g>")
target_compile_options(gpgpu_test PUBLIC "$<$<CONFIG:RELEASE>:-ffast-math>")
target_compile_options(gpgpu_test PUBLIC "$<$<CONFIG:RELEASE>:-O3>")
target_compile_options(gpgpu_test PUBLIC "$<$<CONFIG:RELEASE>:-march=native>")

# This is OSX specific
target_link_libraries(gpgpu_test "-framework OpenCL")
