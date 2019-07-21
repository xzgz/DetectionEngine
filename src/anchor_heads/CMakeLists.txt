

file(GLOB anchor_srcs ${PROJECT_SOURCE_DIR}/src/anchor_heads/*.cc)

add_library(anchor_heads ${anchor_srcs})
target_link_libraries(anchor_heads PUBLIC util ${CObjectFlow_LINKER_LIBS})
