

file(GLOB core_srcs ${PROJECT_SOURCE_DIR}/src/core/*/*.cc)

add_library(core ${core_srcs})
target_link_libraries(core PUBLIC ${CObjectFlow_LINKER_LIBS})
