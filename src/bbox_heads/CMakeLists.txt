
file(GLOB bbox_heads_srcs ${PROJECT_SOURCE_DIR}/src/bbox_heads/*.cc)

add_library(bbox_heads ${bbox_heads_srcs})
target_link_libraries(bbox_heads PUBLIC util ${CObjectFlow_LINKER_LIBS})
