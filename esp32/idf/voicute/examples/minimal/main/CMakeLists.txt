idf_component_register(
    # C++ required: voicute.h pulls in TFLite Micro C++ headers
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES voicute spiffs
)
