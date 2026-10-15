/* Shared by the PiPL resource (.r, preprocessed by cl /EP) and the C++ code. Numeric literals only:
 * PiPLTool cannot evaluate expressions. DMS_Depth.cpp static_asserts these against the SDK enums so
 * the PiPL and PF_Cmd_GLOBAL_SETUP flags can never drift apart. */
#ifndef DMS_VERSION_H
#define DMS_VERSION_H

#define DMS_MAJOR_VERSION 1
#define DMS_MINOR_VERSION 0
#define DMS_BUG_VERSION 0
#define DMS_STAGE_VERSION 0 /* PF_Stage_DEVELOP */
#define DMS_BUILD_VERSION 1

/* PF_VERSION(1,0,0,PF_Stage_DEVELOP,1) = (1<<19) | 1 */
#define DMS_PIPL_EFFECT_VERSION 524289

/* PF_OutFlag_WIDE_TIME_INPUT (1<<1) | PF_OutFlag_NON_PARAM_VARY (1<<2) | PF_OutFlag_I_DO_DIALOG (1<<5)
 * | PF_OutFlag_DEEP_COLOR_AWARE (1<<25) | PF_OutFlag_SEND_UPDATE_PARAMS_UI (1<<26)
 * NON_PARAM_VARY is cleared dynamically (PF_Cmd_QUERY_DYNAMIC_FLAGS) unless the temporal model is selected. */
#define DMS_PIPL_OUTFLAGS 0x06000026

/* PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS (1<<0) | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG (1<<3)
 * | PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10) | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12)
 * | PF_OutFlag2_SUPPORTS_GET_FLATTENED_SEQUENCE_DATA (1<<23) | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27) */
#define DMS_PIPL_OUTFLAGS2 0x08801409

#define DMS_EFFECT_NAME "DMS Depth Map"
#define DMS_EFFECT_MATCH_NAME "DMS DepthMap AI"
#define DMS_EFFECT_CATEGORY "DMS"
#define DMS_SUPPORT_URL "https://anchorpoint.studio"

#endif
