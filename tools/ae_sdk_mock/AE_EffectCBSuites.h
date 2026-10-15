#pragma once
#include "AE_Effect.h"
#define kPFHandleSuite "PF Handle Suite"
#define kPFHandleSuiteVersion1 1
typedef struct PF_HandleSuite1 {
    PF_Handle (*host_new_handle)(A_HandleSize);
    void* (*host_lock_handle)(PF_Handle);
    void (*host_unlock_handle)(PF_Handle);
    void (*host_dispose_handle)(PF_Handle);
    A_HandleSize (*host_get_handle_size)(PF_Handle);
    PF_Err (*host_resize_handle)(A_HandleSize, PF_Handle*);
} PF_HandleSuite1;
#define kPFWorldSuite "PF World Suite"
#define kPFWorldSuiteVersion2 2
typedef struct PF_WorldSuite2 {
    PF_Err (*PF_NewWorld)(PF_ProgPtr, A_long, A_long, PF_Boolean, PF_PixelFormat, PF_EffectWorld*);
    PF_Err (*PF_DisposeWorld)(PF_ProgPtr, PF_EffectWorld*);
    PF_Err (*PF_GetPixelFormat)(const PF_EffectWorld*, PF_PixelFormat*);
} PF_WorldSuite2;
#define kPFEffectSequenceDataSuite "PF Effect Sequence Data Suite"
#define kPFEffectSequenceDataSuiteVersion1 1
typedef struct PF_EffectSequenceDataSuite1 { PF_Err (*PF_GetConstSequenceData)(PF_ProgPtr, PF_ConstHandle*); } PF_EffectSequenceDataSuite1;
#define kPFParamUtilsSuite "PF Param Utils Suite"
#define kPFParamUtilsSuiteVersion3 3
typedef struct PF_ParamUtilsSuite3 { PF_Err (*PF_UpdateParamUI)(PF_ProgPtr, PF_ParamIndex, const PF_ParamDef*); } PF_ParamUtilsSuite3;
