// All After Effects SDK includes in one place (compiled as external/system headers).
#pragma once

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_GeneralPlug.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHelper.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"
#include "String_Utils.h"

// Suite versions used by the plugin. Centralized so an SDK bump is a one-line change.
// These are long-standing versions shipped by every AE release since CC 2019.
using DmsHandleSuite = PF_HandleSuite1;
#define DMS_HANDLE_SUITE kPFHandleSuite, kPFHandleSuiteVersion1
using DmsWorldSuite = PF_WorldSuite2;
#define DMS_WORLD_SUITE kPFWorldSuite, kPFWorldSuiteVersion2
using DmsSeqDataSuite = PF_EffectSequenceDataSuite1;
#define DMS_SEQDATA_SUITE kPFEffectSequenceDataSuite, kPFEffectSequenceDataSuiteVersion1
using DmsParamUtilsSuite = PF_ParamUtilsSuite3;
#define DMS_PARAMUTILS_SUITE kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3
using DmsUtilitySuite = AEGP_UtilitySuite6;
#define DMS_UTILITY_SUITE kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6
using DmsPFInterfaceSuite = AEGP_PFInterfaceSuite1;
#define DMS_PFINTERFACE_SUITE kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1
using DmsStreamSuite = AEGP_StreamSuite5;
#define DMS_STREAM_SUITE kAEGPStreamSuite, kAEGPStreamSuiteVersion5
using DmsDynStreamSuite = AEGP_DynamicStreamSuite4;
#define DMS_DYNSTREAM_SUITE kAEGPDynamicStreamSuite, kAEGPDynamicStreamSuiteVersion4
using DmsEffectSuite = AEGP_EffectSuite4;
#define DMS_EFFECT_SUITE kAEGPEffectSuite, kAEGPEffectSuiteVersion4
