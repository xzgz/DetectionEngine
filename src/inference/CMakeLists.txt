
file(GLOB inference_srcs       ${PROJECT_SOURCE_DIR}/src/inference/*.cc)

add_library(inference ${inference_srcs})
target_link_libraries(inference PUBLIC detectors ${CObjectFlow_LINKER_LIBS})
