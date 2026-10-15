#pragma once
#include "AE_Effect.h"
typedef A_long AEGP_PluginID;
typedef struct _AEGP_EffectRefH* AEGP_EffectRefH;
typedef struct _AEGP_StreamRefH* AEGP_StreamRefH;
enum { AEGP_DynStreamFlag_ACTIVE_EYEBALL = 0, AEGP_DynStreamFlag_HIDDEN = 1, AEGP_DynStreamFlag_DISABLED = 2 };
typedef A_long AEGP_DynStreamFlags;
#define kAEGPUtilitySuite "AEGP Utility Suite"
#define kAEGPUtilitySuiteVersion6 6
typedef struct AEGP_UtilitySuite6 { A_Err (*AEGP_RegisterWithAEGP)(void*, const A_char*, AEGP_PluginID*); } AEGP_UtilitySuite6;
#define kAEGPPFInterfaceSuite "AEGP PF Interface Suite"
#define kAEGPPFInterfaceSuiteVersion1 1
typedef struct AEGP_PFInterfaceSuite1 { A_Err (*AEGP_GetNewEffectForEffect)(AEGP_PluginID, PF_ProgPtr, AEGP_EffectRefH*); } AEGP_PFInterfaceSuite1;
#define kAEGPStreamSuite "AEGP Stream Suite"
#define kAEGPStreamSuiteVersion5 5
typedef struct AEGP_StreamSuite5 {
    A_Err (*AEGP_GetNewEffectStreamByIndex)(AEGP_PluginID, AEGP_EffectRefH, PF_ParamIndex, AEGP_StreamRefH*);
    A_Err (*AEGP_DisposeStream)(AEGP_StreamRefH);
} AEGP_StreamSuite5;
#define kAEGPDynamicStreamSuite "AEGP Dynamic Stream Suite"
#define kAEGPDynamicStreamSuiteVersion4 4
typedef struct AEGP_DynamicStreamSuite4 { A_Err (*AEGP_SetDynamicStreamFlag)(AEGP_StreamRefH, AEGP_DynStreamFlags, A_long, A_long); } AEGP_DynamicStreamSuite4;
#define kAEGPEffectSuite "AEGP Effect Suite"
#define kAEGPEffectSuiteVersion4 4
typedef struct AEGP_EffectSuite4 { A_Err (*AEGP_DisposeEffect)(AEGP_EffectRefH); } AEGP_EffectSuite4;
