/*
 * head.h — 唤醒词模型 MultiProto head 权重
 *
 * 此文件由模型训练脚本自动生成, 请用你自己的 head.h 替换。
 *
 * 生成方式:
 *   运行训练导出脚本, 会生成包含以下常量的 head.h:
 *
 *     KWS_HEAD_K       原型数量 (通常 3-7)
 *     KWS_HEAD_D       特征维度 (通常 256)
 *     KWS_ABS_TEMP     温度参数的倒数 (1/|temp|)
 *     KWS_FC_B         线性分类器偏置
 *     KWS_FC_W[K]      线性分类器权重
 *     KWS_PROTO_NORM[K][D]  L2归一化后的原型向量
 *
 * 注意: head.h 必须和 .tflite 模型配套使用 (同一轮训练导出)
 *
 * 当前是一个占位示例 — 请替换为你自己的权重!
 */

#pragma once
#define KWS_HEAD_K 5
#define KWS_HEAD_D 256

// ↓ 以下为示例值, 请替换 ↓
static const float KWS_ABS_TEMP = 0.1f;
static const float KWS_FC_B = 0.0f;
static const float KWS_FC_W[5] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float KWS_PROTO_NORM[5][256] = { { 0.0f } };
