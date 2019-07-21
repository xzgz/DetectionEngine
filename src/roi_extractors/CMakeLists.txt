
file(GLOB roi_extractors_srcs ${PROJECT_SOURCE_DIR}/src/roi_extractors/*.cc)

add_library(roi_extractors ${roi_extractors_srcs})
target_link_libraries(roi_extractors PUBLIC util ${CObjectFlow_LINKER_LIBS})
