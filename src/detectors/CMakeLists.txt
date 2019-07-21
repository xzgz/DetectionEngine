
file(GLOB detector_srcs ${PROJECT_SOURCE_DIR}/src/detectors/*.cc)

add_library(detectors ${detector_srcs})
target_link_libraries(detectors PUBLIC anchor_heads bbox_heads roi_extractors
        core ${CObjectFlow_LINKER_LIBS})
