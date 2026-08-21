#include "afe_processor.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "model_path.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "AFE_PROC";

struct afe_processor {
    const esp_afe_sr_iface_t *handle;
    esp_afe_sr_data_t        *data;
    int                       feed_chunksize;
    int                       feed_nch;
    int16_t                  *fetch_buf;
    int                       fetch_buf_size;
};

afe_processor_t *afe_processor_init(const afe_processor_config_t *cfg) {
    afe_processor_config_t defaults = {
        .input_format = "RMNM",
        .afe_type = 0,
        .afe_mode = 0,
        .wakenet_init = false,
        .vad_init = false,
        .ns_init = false,
    };
    if (cfg) {
        if (cfg->input_format) defaults.input_format = cfg->input_format;
        defaults.afe_type = cfg->afe_type;
        defaults.afe_mode = cfg->afe_mode;
        defaults.wakenet_init = cfg->wakenet_init;
        defaults.vad_init = cfg->vad_init;
        defaults.ns_init = cfg->ns_init;
    }

    afe_processor_t *ap = calloc(1, sizeof(*ap));
    if (!ap) return NULL;

    srmodel_list_t *models = esp_srmodel_init("model");

    afe_config_t *afe_cfg = afe_config_init(defaults.input_format, models,
                                            defaults.afe_type, defaults.afe_mode);
    if (!afe_cfg) { free(ap); return NULL; }

    afe_cfg->wakenet_init = defaults.wakenet_init;
    afe_cfg->vad_init = defaults.vad_init;
    afe_cfg->ns_init = defaults.ns_init;
    if (defaults.ns_init) {
        afe_cfg->ns_model_name = "webrtc_ns";  // No model files needed!
        afe_cfg->afe_ns_mode = AFE_NS_MODE_WEBRTC;
    }

    ap->handle = esp_afe_handle_from_config(afe_cfg);
    if (!ap->handle) { afe_config_free(afe_cfg); free(ap); return NULL; }

    ap->data = ap->handle->create_from_config(afe_cfg);
    afe_config_free(afe_cfg);
    if (!ap->data) { free(ap); return NULL; }

    ap->feed_chunksize = ap->handle->get_feed_chunksize(ap->data);
    ap->feed_nch = ap->handle->get_feed_channel_num(ap->data);

    ESP_LOGI(TAG, "Init: format=%s feed=%d ch=%d",
             defaults.input_format, ap->feed_chunksize, ap->feed_nch);

    return ap;
}

void afe_processor_feed(afe_processor_t *ap, const int16_t *data, int len) {
    if (!ap) return;
    // Cast away const — AFE feed takes non-const but doesn't modify
    ap->handle->feed(ap->data, (int16_t*)data);
}

int afe_processor_fetch(afe_processor_t *ap, int16_t *out, int max_samples) {
    if (!ap) return 0;

    afe_fetch_result_t *res = ap->handle->fetch(ap->data);
    if (!res || res->ret_value != ESP_OK || !res->data || res->data_size <= 0)
        return 0;

    int available = res->data_size / sizeof(int16_t);
    int copy = (available < max_samples) ? available : max_samples;
    memcpy(out, res->data, copy * sizeof(int16_t));
    return copy;
}

int afe_processor_feed_chunksize(afe_processor_t *ap) {
    return ap ? ap->feed_chunksize : 0;
}

int afe_processor_feed_channels(afe_processor_t *ap) {
    return ap ? ap->feed_nch : 0;
}

void afe_processor_deinit(afe_processor_t *ap) {
    if (!ap) return;
    // AFE data lifecycle: created with create_from_config, no explicit destroy in public API
    free(ap);
}
