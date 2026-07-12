idf_component_register(
    SRCS
        "model_loader.cpp"
        "recognizer.cpp"
        "mel_extractor.c"
        "detect_logic.c"
    INCLUDE_DIRS "."
    REQUIRES spiffs esp_timer
)
