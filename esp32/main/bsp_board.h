#pragma once

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/i2s_tdm.h"
#include "driver/i2c_master.h"

#include "soc/soc_caps.h"
#include "esp_idf_version.h"

#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_codec_dev_os.h"
#include "sdkconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FILE_NAME_SIZE 100  // Define maximum file name size
#define MAX_PATH_SIZE 512      // Define a larger size for the full path

/**
 * @brief ESP32-S3-CAM-OVxxxx I2C GPIO defineation
 * 
 */
#define I2C_NUM         (0)
#define GPIO_I2C_SCL    (GPIO_NUM_10)
#define GPIO_I2C_SDA    (GPIO_NUM_11)

/**
 * @brief ESP32-S3-CAM-OVxxxx SDMMC GPIO defination
 * 
 * @note Only avaliable when PMOD connected
 */
#define FUNC_SDMMC_EN   (1)
#define SDMMC_BUS_WIDTH (1)
#define GPIO_SDMMC_CLK  (GPIO_NUM_40)
#define GPIO_SDMMC_CMD  (GPIO_NUM_42)
#define GPIO_SDMMC_D0   (GPIO_NUM_41)
#define GPIO_SDMMC_D1   (GPIO_NUM_NC)
#define GPIO_SDMMC_D2   (GPIO_NUM_NC)
#define GPIO_SDMMC_D3   (GPIO_NUM_NC)
#define GPIO_SDMMC_DET  (GPIO_NUM_NC)

/**
 * @brief ESP32-S3-CAM-OVxxxx SDSPI GPIO definationv
 * 
 */
#define FUNC_SDSPI_EN       (0)
#define SDSPI_HOST          (SPI2_HOST)
#define GPIO_SDSPI_CS       (GPIO_NUM_NC)
#define GPIO_SDSPI_SCLK     (GPIO_NUM_NC)
#define GPIO_SDSPI_MISO     (GPIO_NUM_NC)
#define GPIO_SDSPI_MOSI     (GPIO_NUM_NC)

/**
 * @brief ESP32-S3-CAM-OVxxxx I2S GPIO defination
 * 
 */
#define FUNC_I2S_EN         (1)
#define GPIO_I2S_LRCK       (GPIO_NUM_14)
#define GPIO_I2S_MCLK       (GPIO_NUM_12)
#define GPIO_I2S_SCLK       (GPIO_NUM_13)
#define GPIO_I2S_SDIN       (GPIO_NUM_15)
#define GPIO_I2S_DOUT       (GPIO_NUM_16)

/**
 * @brief ESP32-S3-CAM-OVxxxx I2S GPIO defination
 * 
 */
#define FUNC_I2S0_EN         (0)
#define GPIO_I2S0_LRCK       (GPIO_NUM_NC)
#define GPIO_I2S0_MCLK       (GPIO_NUM_NC)
#define GPIO_I2S0_SCLK       (GPIO_NUM_NC)
#define GPIO_I2S0_SDIN       (GPIO_NUM_NC)
#define GPIO_I2S0_DOUT       (GPIO_NUM_NC)

/**
 * @brief record configurations
 *
 */
#define RECORD_VOLUME   (36.0)

/**
 * @brief player configurations
 *
 */
#define PLAYER_VOLUME   (60)

/**
 * @brief ESP32-S3-HMI-DevKit power control IO
 * 
 * @note Some power control pins might not be listed yet
 * 
 */
//#define FUNC_PWR_CTRL       (1)
#define GPIO_PWR_CTRL       (-1)
#define GPIO_PWR_ON_LEVEL   (1)

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)

#define I2S_CONFIG_DEFAULT(sample_rate, channel_fmt, bits_per_chan) { \
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(16000), \
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(32, I2S_SLOT_MODE_STEREO), \
        .gpio_cfg = { \
            .mclk = GPIO_I2S_MCLK, \
            .bclk = GPIO_I2S_SCLK, \
            .ws   = GPIO_I2S_LRCK, \
            .dout = GPIO_I2S_DOUT, \
            .din  = GPIO_I2S_SDIN, \
        }, \
    }

#else

#define I2S_CONFIG_DEFAULT(sample_rate, channel_fmt, bits_per_chan) { \
    .mode                   = I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX, \
    .sample_rate            = 16000, \
    .bits_per_sample        = I2S_BITS_PER_SAMPLE_32BIT, \
    .channel_format         = I2S_CHANNEL_FMT_RIGHT_LEFT, \
    .communication_format   = I2S_COMM_FORMAT_STAND_I2S, \
    .intr_alloc_flags       = ESP_INTR_FLAG_LEVEL1, \
    .dma_buf_count          = 6, \
    .dma_buf_len            = 160, \
    .use_apll               = false, \
    .tx_desc_auto_clear     = true, \
    .fixed_mclk             = 0, \
    .mclk_multiple          = I2S_MCLK_MULTIPLE_DEFAULT, \
    .bits_per_chan          = I2S_BITS_PER_CHAN_32BIT, \
}

#endif


/* LCD settings */
#define EXAMPLE_LCD_SPI_NUM         (SPI3_HOST)
#define EXAMPLE_LCD_PIXEL_CLK_HZ    (80 * 1000 * 1000)
#define EXAMPLE_LCD_CMD_BITS        (8)
#define EXAMPLE_LCD_PARAM_BITS      (8)
#define EXAMPLE_LCD_BITS_PER_PIXEL  (16)
#define EXAMPLE_LCD_DRAW_BUFF_DOUBLE (1)
#define EXAMPLE_LCD_DRAW_BUFF_HEIGHT (50)
#define EXAMPLE_LCD_BL_ON_LEVEL     (1)
#define Backlight_MAX           100   
#define DEFAULT_BACKLIGHT       80 

/* LCD pins */
#define EXAMPLE_LCD_GPIO_SCLK       (GPIO_NUM_5)
#define EXAMPLE_LCD_GPIO_MOSI       (GPIO_NUM_1)
#define EXAMPLE_LCD_GPIO_RST        (GPIO_NUM_NC)
#define EXAMPLE_LCD_GPIO_DC         (GPIO_NUM_3)
#define EXAMPLE_LCD_GPIO_CS         (GPIO_NUM_6)
#define EXAMPLE_LCD_GPIO_BL         (GPIO_NUM_NC)

/* LCD touch pins */
#define EXAMPLE_TOUCH_GPIO_RST      (GPIO_NUM_NC)

#ifdef CONFIG_WAVESHARE_1_47INCH_TOUCH_LCD
#define EXAMPLE_LCD_H_RES   (172)
#define EXAMPLE_LCD_V_RES   (320)
#define EXAMPLE_TOUCH_GPIO_INT      (GPIO_NUM_9)
#elif defined(CONFIG_WAVESHARE_2INCH_TOUCH_LCD)
#define EXAMPLE_LCD_H_RES   (240)
#define EXAMPLE_LCD_V_RES   (320)
#define EXAMPLE_TOUCH_GPIO_INT      (GPIO_NUM_9)
#elif defined(CONFIG_WAVESHARE_2_8INCH_TOUCH_LCD)
#define EXAMPLE_LCD_H_RES   (240)
#define EXAMPLE_LCD_V_RES   (320)
#define EXAMPLE_TOUCH_GPIO_INT      (GPIO_NUM_NC)
#elif defined(CONFIG_WAVESHARE_3_5INCH_TOUCH_LCD)
#define EXAMPLE_LCD_H_RES   (320)
#define EXAMPLE_LCD_V_RES   (480)
#define EXAMPLE_TOUCH_GPIO_INT      (GPIO_NUM_9)
#endif

// GPIO assignment
#define LED_STRIP_GPIO_PIN  38
#define LED_STRIP_LED_COUNT 7


esp_err_t esp_board_init(uint32_t sample_rate, int channel_format, int bits_per_chan);

esp_err_t esp_sdcard_init(char *mount_point, size_t max_files);
esp_err_t esp_sdcard_deinit(char *mount_point);

esp_err_t esp_audio_play(const int16_t* data, int length, uint32_t ticks_to_wait);

esp_err_t esp_get_feed_data(bool is_get_raw_channel, int16_t *buffer, int buffer_len);
int esp_get_feed_channel(void);
char* esp_get_input_format(void);
esp_err_t esp_audio_set_play_vol(int volume);
esp_err_t esp_audio_get_play_vol(int *volume);

i2c_master_bus_handle_t esp_ret_i2c_handle(void);
esp_codec_dev_handle_t esp_ret_play_dev();
uint32_t Get_SD_Size(void);
uint16_t Folder_retrieval(const char* directory, const char* fileExtension, char File_Name[][MAX_FILE_NAME_SIZE], uint16_t maxFiles) ;

#ifdef __cplusplus
}
#endif
