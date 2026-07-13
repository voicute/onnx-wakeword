"""Patch led_strip SPI driver for ESP-IDF v6.x compatibility."""
import os

target = 'managed_components/espressif__led_strip/src/led_strip_spi_dev.c'
if not os.path.exists(target):
    exit(0)

with open(target, 'r') as f:
    content = f.read()

if 'esp_heap_caps.h' in content:
    exit(0)  # already patched

old = '#include "esp_check.h"'
new = '#include "esp_check.h"\n#include "esp_heap_caps.h"'
with open(target, 'w') as f:
    f.write(content.replace(old, new))

print('[Voicute] led_strip patched for ESP-IDF v6.x')
