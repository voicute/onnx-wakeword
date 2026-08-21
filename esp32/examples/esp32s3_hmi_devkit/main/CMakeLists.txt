set(SOURCES
    "./main.cpp"
    "./bsp_board.c"
)

set(INCLUDE_DIRS ".")

idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS ${INCLUDE_DIRS}
    REQUIRES voicute afe_processor esp_driver_gpio esp_driver_i2s esp_driver_i2c fatfs sdmmc
)
