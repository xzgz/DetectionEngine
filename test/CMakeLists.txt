# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} PUBLIC inference ${CObjectFlow_LINKER_LIBS})

endforeach(source)
