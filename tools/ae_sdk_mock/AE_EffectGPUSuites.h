#pragma once
#include "AE_Effect.h"
#define kPFGPUDeviceSuite "PF GPU Device Suite"
#define kPFGPUDeviceSuiteVersion1 1
typedef struct { PF_GPU_Framework device_framework; PF_Boolean compatibleB; void* platformPV; void* devicePV; void* contextPV; void* command_queuePV;
                 void* offscreen_opengl_contextPV; void* offscreen_opengl_devicePV; } PF_GPUDeviceInfo;
typedef struct PF_GPUDeviceSuite1 {
    PF_Err (*GetDeviceCount)(PF_ProgPtr, A_u_long*);
    PF_Err (*GetDeviceInfo)(PF_ProgPtr, A_u_long, PF_GPUDeviceInfo*);
    PF_Err (*GetGPUWorldData)(PF_ProgPtr, PF_EffectWorld*, void**);
} PF_GPUDeviceSuite1;
