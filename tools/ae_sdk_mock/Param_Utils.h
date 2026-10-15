#pragma once
#include "AE_Effect.h"
#include "AE_Macros.h"
#include <cstring>
#define PF_STRCPY(DST, SRC) std::strncpy((DST), (SRC), sizeof(DST) - 1)
#define PF_ADD_POPUPX(NAME, NUMCHOICES, DFLT, ITEMSTR, FLAGS, ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_POPUP; \
    PF_STRCPY(def.name, NAME); def.u.pd.num_choices = (NUMCHOICES); def.u.pd.dephault = (DFLT); def.u.pd.value = def.u.pd.dephault; \
    def.u.pd.u.namesptr = (ITEMSTR); def.flags = (FLAGS); def.uu.id = (ID); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_ADD_FLOAT_SLIDERX(NAME, VMIN, VMAX, SMIN, SMAX, DFLT, PREC, DISP, FLAGS, ID) do { PF_Err priv_err = PF_Err_NONE; \
    def.param_type = PF_Param_FLOAT_SLIDER; PF_STRCPY(def.name, NAME); def.u.fs_d.valid_min = (VMIN); def.u.fs_d.slider_min = (SMIN); \
    def.u.fs_d.valid_max = (VMAX); def.u.fs_d.slider_max = (SMAX); def.u.fs_d.value = (DFLT); def.u.fs_d.dephault = (DFLT); \
    def.u.fs_d.precision = (PREC); def.u.fs_d.display_flags = (DISP); def.flags |= (FLAGS); def.uu.id = (ID); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_ADD_SLIDER(NAME, VMIN, VMAX, SMIN, SMAX, DFLT, ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_SLIDER; \
    PF_STRCPY(def.name, NAME); def.u.sd.value = def.u.sd.dephault = (DFLT); def.uu.id = (ID); (void)(VMIN); (void)(VMAX); (void)(SMIN); (void)(SMAX); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_ADD_CHECKBOXX(NAME, DFLT, FLAGS, ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_CHECKBOX; \
    PF_STRCPY(def.name, NAME); def.u.bd.value = def.u.bd.dephault = (DFLT); def.flags |= (FLAGS); def.uu.id = (ID); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_ADD_POINT(NAME, X_DFLT, Y_DFLT, RESTRICT_BOUNDS, ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_POINT; \
    PF_STRCPY(def.name, NAME); def.u.td.restrict_bounds = (RESTRICT_BOUNDS); def.u.td.x_dephault = (PF_Fixed)((X_DFLT) * 65536); \
    def.u.td.y_dephault = (PF_Fixed)((Y_DFLT) * 65536); def.uu.id = (ID); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_ADD_TOPIC(NAME, ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_GROUP_START; PF_STRCPY(def.name, NAME); \
    def.uu.id = (ID); if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
#define PF_END_TOPIC(ID) do { PF_Err priv_err = PF_Err_NONE; def.param_type = PF_Param_GROUP_END; def.uu.id = (ID); \
    if ((priv_err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE) return priv_err; } while (0)
