set(SOURCES
    "./main.cpp"
    "./bsp_board.c"
)

set(INCLUDE_DIRS ".")

idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS ${INCLUDE_DIRS}
    REQUIRES voicute
)
