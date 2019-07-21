
file(GLOB util_srcs ${PROJECT_SOURCE_DIR}/src/utils/*.cc)

add_library(util ${util_srcs})
target_link_libraries(util PUBLIC ${CObjectFlow_LINKER_LIBS})
