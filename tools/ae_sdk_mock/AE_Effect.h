// COMPILE-CHECK MOCK of the AE SDK subset used by DMS (see README.md). Not the Adobe SDK.
#pragma once
#include <cstddef>
#include <cstdint>

typedef int32_t A_long;
typedef uint32_t A_u_long;
typedef int16_t A_short;
typedef char A_char;
typedef uint8_t A_u_char;
typedef uint16_t A_u_short;
typedef int32_t A_Err;
typedef uint64_t A_HandleSize;
typedef A_long PF_Err;
typedef void** PF_Handle;
typedef const void* const* PF_ConstHandle;
typedef struct _PF_ProgressInfo* PF_ProgPtr;
typedef A_long PF_Fixed;
typedef double PF_FpLong;
typedef float PF_FpShort;
typedef unsigned char PF_Boolean;
typedef A_long PF_ParamIndex;
typedef A_long PF_ParamFlags;
typedef A_long PF_ParamUIFlags;
typedef A_long PF_ParamValue;
typedef A_long PF_OutFlags;
typedef A_long PF_OutFlags2;
typedef A_long PF_Cmd;
typedef void* PF_PixelPtr;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define PF_MAX_EFFECT_MSG_LEN 255
#define PF_MAX_EFFECT_PARAM_NAME_LEN 31
#define DllExport
#define SPAPI
#define AE_RESERVED_INFO 8

enum { PF_Err_NONE = 0, PF_Err_OUT_OF_MEMORY = 4, PF_Err_INTERNAL_STRUCT_DAMAGED = 512, PF_Err_INVALID_INDEX, PF_Err_UNRECOGNIZED_PARAM_TYPE,
       PF_Err_INVALID_CALLBACK, PF_Err_BAD_CALLBACK_PARAM, PF_Interrupt_CANCEL };
enum { A_Err_NONE = 0 };

enum {
    PF_Cmd_ABOUT = 0, PF_Cmd_GLOBAL_SETUP, PF_Cmd_UNUSED_0, PF_Cmd_GLOBAL_SETDOWN, PF_Cmd_PARAMS_SETUP, PF_Cmd_SEQUENCE_SETUP,
    PF_Cmd_SEQUENCE_RESETUP, PF_Cmd_SEQUENCE_FLATTEN, PF_Cmd_SEQUENCE_SETDOWN, PF_Cmd_DO_DIALOG, PF_Cmd_FRAME_SETUP, PF_Cmd_RENDER,
    PF_Cmd_FRAME_SETDOWN, PF_Cmd_USER_CHANGED_PARAM, PF_Cmd_UPDATE_PARAMS_UI, PF_Cmd_EVENT, PF_Cmd_GET_EXTERNAL_DEPENDENCIES,
    PF_Cmd_COMPLETELY_GENERAL, PF_Cmd_QUERY_DYNAMIC_FLAGS, PF_Cmd_AUDIO_RENDER, PF_Cmd_AUDIO_SETUP, PF_Cmd_AUDIO_SETDOWN,
    PF_Cmd_ARBITRARY_CALLBACK, PF_Cmd_SMART_PRE_RENDER, PF_Cmd_SMART_RENDER, PF_Cmd_RESERVED1, PF_Cmd_RESERVED2, PF_Cmd_RESERVED3,
    PF_Cmd_GET_FLATTENED_SEQUENCE_DATA, PF_Cmd_TRANSLATE_PARAMS_TO_PREFS, PF_Cmd_RESERVED4, PF_Cmd_SMART_RENDER_GPU, PF_Cmd_GPU_DEVICE_SETUP,
    PF_Cmd_GPU_DEVICE_SETDOWN
};

enum {
    PF_OutFlag_NONE = 0, PF_OutFlag_WIDE_TIME_INPUT = 1L << 1, PF_OutFlag_NON_PARAM_VARY = 1L << 2, PF_OutFlag_I_DO_DIALOG = 1L << 5,
    PF_OutFlag_DISPLAY_ERROR_MESSAGE = 1L << 8, PF_OutFlag_FORCE_RERENDER = 1L << 22, PF_OutFlag_DEEP_COLOR_AWARE = 1L << 25,
    PF_OutFlag_SEND_UPDATE_PARAMS_UI = 1L << 26
};
enum {
    PF_OutFlag2_NONE = 0, PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS = 1L << 0, PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG = 1L << 3,
    PF_OutFlag2_SUPPORTS_SMART_RENDER = 1L << 10, PF_OutFlag2_FLOAT_COLOR_AWARE = 1L << 12,
    PF_OutFlag2_SUPPORTS_GET_FLATTENED_SEQUENCE_DATA = 1L << 23, PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 = 1L << 25,
    PF_OutFlag2_SUPPORTS_THREADED_RENDERING = 1L << 27
};
enum { PF_ParamFlag_CANNOT_TIME_VARY = 1L << 1, PF_ParamFlag_SUPERVISE = 1L << 4, PF_ParamFlag_START_COLLAPSED = 1L << 5 };
enum { PF_PUI_INVISIBLE = 1L << 9, PF_PUI_DISABLED = 1L << 8 };
enum { PF_Precision_INTEGER, PF_Precision_TENTHS, PF_Precision_HUNDREDTHS, PF_Precision_THOUSANDTHS, PF_Precision_TEN_THOUSANDTHS };
enum { PF_ValueDisplayFlag_NONE = 0, PF_ValueDisplayFlag_PERCENT = 1 };
enum { PF_Stage_DEVELOP, PF_Stage_ALPHA, PF_Stage_BETA, PF_Stage_RELEASE };
typedef enum { PF_PixelFormat_ARGB32 = 'argb', PF_PixelFormat_ARGB64 = 'ar64', PF_PixelFormat_ARGB128 = 'ar28', PF_PixelFormat_GPU_BGRA128 = '8ARG',
               PF_PixelFormat_INVALID = 'badf' } PF_PixelFormat;
typedef enum { PF_GPU_Framework_NONE = 0, PF_GPU_Framework_OPENCL, PF_GPU_Framework_METAL, PF_GPU_Framework_CUDA, PF_GPU_Framework_DIRECTX } PF_GPU_Framework;
enum { PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS = 0x1, PF_RenderOutputFlag_GPU_RENDER_POSSIBLE = 0x2 };
enum { PF_Param_LAYER = 0, PF_Param_SLIDER, PF_Param_FIX_SLIDER, PF_Param_ANGLE, PF_Param_CHECKBOX, PF_Param_COLOR, PF_Param_POINT,
       PF_Param_POPUP, PF_Param_CUSTOM, PF_Param_NO_DATA, PF_Param_FLOAT_SLIDER, PF_Param_GROUP_START = 13, PF_Param_GROUP_END };

#define PF_Vers_BUILD_BITS 0x1ffL
#define PF_Vers_STAGE_BITS 0x3L
#define PF_Vers_BUGFIX_BITS 0xfL
#define PF_Vers_SUBVERS_BITS 0xfL
#define PF_Vers_VERS_BITS 0x7L
#define PF_VERSION(vers, subvers, bugvers, stage, build)                                                                         \
    (A_u_long)((((A_u_long)(vers) & PF_Vers_VERS_BITS) << 19) | (((A_u_long)(subvers) & PF_Vers_SUBVERS_BITS) << 15) |          \
               (((A_u_long)(bugvers) & PF_Vers_BUGFIX_BITS) << 11) | (((A_u_long)(stage) & PF_Vers_STAGE_BITS) << 9) |          \
               ((A_u_long)(build) & PF_Vers_BUILD_BITS))

typedef struct { A_long left, top, right, bottom; } PF_LRect;
typedef struct { A_long num; A_u_long den; } PF_RationalScale;

typedef struct PF_LayerDef {
    void* reserved0;
    void* reserved1;
    A_long world_flags;
    PF_PixelPtr data;
    A_long rowbytes;
    A_long width;
    A_long height;
    PF_LRect extent_hint;
    void* platform_ref;
    A_long reserved_long1;
    void* reserved_long4;
    PF_RationalScale pix_aspect_ratio;
    void* reserved_long2;
    A_long origin_x;
    A_long origin_y;
    A_long reserved_long3;
    A_long dephault;
} PF_LayerDef, PF_EffectWorld;

typedef struct { A_long value; A_char reserved; A_long dephault; } PF_SliderDef;
typedef struct { PF_FpLong value; PF_FpLong phase; void* value_str; A_short precision; PF_FpLong valid_min, valid_max, slider_min, slider_max, dephault;
                 A_short display_flags; A_u_long fs_flags; PF_FpShort curve_tolerance; A_short reserved; } PF_FloatSliderDef;
typedef struct { A_long value; A_long reserved; A_long dephault; const A_char* nameptr; } PF_CheckBoxDef;
typedef struct { A_long value; A_short num_choices; A_short dephault; union { const A_char* namesptr; } u; } PF_PopupDef;
typedef struct { PF_Fixed x_value, y_value; A_char reserved[3]; PF_Boolean restrict_bounds; PF_Fixed x_dephault, y_dephault; } PF_PointDef;
typedef union {
    PF_LayerDef ld;
    PF_SliderDef sd;
    PF_FloatSliderDef fs_d;
    PF_CheckBoxDef bd;
    PF_PopupDef pd;
    PF_PointDef td;
} PF_ParamDefUnion;

typedef struct PF_ParamDef {
    union { A_long id; A_long change_flags; } uu;
    A_long ui_flags;
    A_short ui_width, ui_height;
    A_long param_type;
    A_char name[PF_MAX_EFFECT_PARAM_NAME_LEN + 1];
    PF_ParamFlags flags;
    A_long unused;
    PF_ParamDefUnion u;
} PF_ParamDef;

struct PF_InData;
typedef struct {
    PF_Err (*checkout_param)(PF_ProgPtr, PF_ParamIndex, A_long, A_long, A_u_long, PF_ParamDef*);
    PF_Err (*checkin_param)(PF_ProgPtr, PF_ParamDef*);
    PF_Err (*add_param)(PF_ProgPtr, PF_ParamIndex, PF_ParamDef*);
    PF_Err (*abort)(PF_ProgPtr);
    PF_Err (*progress)(PF_ProgPtr, A_long, A_long);
} PF_InteractCallbacks;
typedef struct { struct { A_long (*sprintf)(A_char*, const A_char*, ...); } ansi; } PF_UtilCallbacks;
struct SPBasicSuite;
typedef struct PF_InData {
    PF_InteractCallbacks inter;
    PF_UtilCallbacks* utils;
    PF_ProgPtr effect_ref;
    A_long quality;
    A_u_long version;
    A_long serial_num;
    A_long appl_id;
    A_long num_params;
    A_long reserved;
    A_long what_cpu, what_fpu;
    A_long current_time, time_step, total_time, local_time_step;
    A_u_long time_scale;
    A_long field;
    PF_Fixed shutter_angle;
    A_long width, height;
    PF_LRect extent_hint;
    A_long output_origin_x, output_origin_y;
    PF_RationalScale downsample_x, downsample_y;
    PF_RationalScale pixel_aspect_ratio;
    A_long in_flags;
    PF_Handle global_data;
    PF_Handle sequence_data;
    PF_Handle frame_data;
    A_long start_sampL, dur_sampL, total_sampL;
    A_long src_snd;
    SPBasicSuite* pica_basicP;
    A_long pre_effect_source_origin_x, pre_effect_source_origin_y;
    PF_Fixed shutter_phase;
} PF_InData;
typedef struct PF_OutData {
    A_u_long my_version;
    A_char name[32];
    PF_Handle global_data;
    A_long num_params;
    PF_Handle sequence_data;
    A_long flat_sdata_size;
    PF_Handle frame_data;
    A_long width, height;
    A_long origin_x, origin_y;
    PF_OutFlags out_flags;
    A_char return_msg[PF_MAX_EFFECT_MSG_LEN + 1];
    A_long start_sampL, dur_sampL;
    A_long dest_snd;
    PF_OutFlags2 out_flags2;
} PF_OutData;

typedef struct { PF_LRect rect; A_long field; A_long channel_mask; PF_Boolean preserve_rgb_of_zero_alpha; char unused[3]; A_long reserved[4]; } PF_RenderRequest;
typedef struct { PF_LRect result_rect; PF_LRect max_result_rect; PF_RationalScale par; A_long solid; PF_Boolean reserved_padB[3]; A_long ref_width, ref_height;
                 A_long reserved[6]; } PF_CheckoutResult;
typedef void (*PF_DeletePreRenderDataFunc)(void*);
typedef struct { PF_RenderRequest output_request; short bitdepth; const void* gpu_data; PF_GPU_Framework what_gpu; A_u_long device_index; } PF_PreRenderInput;
typedef struct { PF_LRect result_rect; PF_LRect max_result_rect; PF_Boolean solid, reserved; A_short flags; void* pre_render_data;
                 PF_DeletePreRenderDataFunc delete_pre_render_data_func; } PF_PreRenderOutput;
typedef struct {
    PF_Err (*checkout_layer)(PF_ProgPtr, PF_ParamIndex, A_long, const PF_RenderRequest*, A_long, A_long, A_u_long, PF_CheckoutResult*);
    PF_Err (*GuidMixInPtr)(PF_ProgPtr, A_u_long, const void*);
} PF_PreRenderCallbacks;
typedef struct { PF_PreRenderInput* input; PF_PreRenderOutput* output; PF_PreRenderCallbacks* cb; } PF_PreRenderExtra;
typedef struct { A_long bitdepth; void* pre_render_data; const void* gpu_data; PF_GPU_Framework what_gpu; A_u_long device_index; } PF_SmartRenderInput;
typedef struct {
    PF_Err (*checkout_layer_pixels)(PF_ProgPtr, A_long, PF_EffectWorld**);
    PF_Err (*checkin_layer_pixels)(PF_ProgPtr, A_long);
    PF_Err (*checkout_output)(PF_ProgPtr, PF_EffectWorld**);
} PF_SmartRenderCallbacks;
typedef struct { PF_SmartRenderInput* input; PF_SmartRenderCallbacks* cb; } PF_SmartRenderExtra;
typedef struct { PF_ParamIndex param_index; } PF_UserChangedParamExtra;
typedef struct { PF_GPU_Framework what_gpu; A_u_long device_index; } PF_GPUDeviceSetupInput;
typedef struct { void* gpu_data; } PF_GPUDeviceSetupOutput;
typedef struct { PF_GPUDeviceSetupInput* input; PF_GPUDeviceSetupOutput* output; } PF_GPUDeviceSetupExtra;
typedef struct { void* gpu_data; PF_GPU_Framework what_gpu; A_u_long device_index; } PF_GPUDeviceSetdownInput;
typedef struct { PF_GPUDeviceSetdownInput* input; } PF_GPUDeviceSetdownExtra;

typedef struct PF_PluginData* PF_PluginDataPtr;
typedef PF_Err (*PF_PluginDataCB2)(PF_PluginDataPtr, const A_u_char*, const A_u_char*, const A_u_char*, const A_u_char*, A_long, A_long, A_long,
                                   A_long, const A_u_char*);
#define PF_REGISTER_EFFECT_EXT2(INPTR, CBPTR, NAME, MATCHNAME, CATEGORY, RESERVEDINFO, ENTRYPOINT, URL)                                   \
    (*(CBPTR))((INPTR), reinterpret_cast<const A_u_char*>(NAME), reinterpret_cast<const A_u_char*>(MATCHNAME),                         \
               reinterpret_cast<const A_u_char*>(CATEGORY), reinterpret_cast<const A_u_char*>(ENTRYPOINT), 0x65464B54 /*'eFKT'*/, 13, 28, \
               (RESERVEDINFO), reinterpret_cast<const A_u_char*>(URL))

#define PF_ABORT(IN_DATA) (*(IN_DATA)->inter.abort)((IN_DATA)->effect_ref)
#define PF_PROGRESS(IN_DATA, CURR, TOTAL) (*(IN_DATA)->inter.progress)((IN_DATA)->effect_ref, (CURR), (TOTAL))
#define PF_CHECKOUT_PARAM(IN_DATA, I, T, STEP, SCALE, PARAM) (*(IN_DATA)->inter.checkout_param)((IN_DATA)->effect_ref, (I), (T), (STEP), (SCALE), (PARAM))
#define PF_CHECKIN_PARAM(IN_DATA, PARAM) (*(IN_DATA)->inter.checkin_param)((IN_DATA)->effect_ref, (PARAM))
#define PF_ADD_PARAM(IN_DATA, INDEX, DEF) (*(IN_DATA)->inter.add_param)((IN_DATA)->effect_ref, (INDEX), (DEF))
