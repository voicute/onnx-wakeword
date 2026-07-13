cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(voicute_wakeword)

# 关闭 TFLite DCHECK 断言 (esp-nn 内核会触发误报)
add_compile_definitions(NDEBUG)

# SPIFFS 镜像 — 存放唤醒词模型 (.tflite)
# 把模型文件放到 spiffs_content/ 目录下即可自动打包
spiffs_create_partition_image(models spiffs_content FLASH_IN_PROJECT)
