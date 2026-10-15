#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "dms_version.h"

#ifndef AE_OS_WIN
	#include <AE_General.r>
#endif

resource 'PiPL' (16000) {
	{
		Kind {
			AEEffect
		},
		Name {
			DMS_EFFECT_NAME
		},
		Category {
			DMS_EFFECT_CATEGORY
		},
#ifdef AE_OS_WIN
	#ifdef AE_PROC_INTELx64
		CodeWin64X86 {"EffectMain"},
	#elif defined(AE_PROC_ARM64)
		CodeWinARM64 {"EffectMain"},
	#endif
#elif defined(AE_OS_MAC)
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		AE_PiPL_Version {
			2,
			0
		},
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		AE_Effect_Version {
			DMS_PIPL_EFFECT_VERSION
		},
		AE_Effect_Info_Flags {
			0
		},
		AE_Effect_Global_OutFlags {
			DMS_PIPL_OUTFLAGS
		},
		AE_Effect_Global_OutFlags_2 {
			DMS_PIPL_OUTFLAGS2
		},
		AE_Effect_Match_Name {
			DMS_EFFECT_MATCH_NAME
		},
		AE_Reserved_Info {
			0
		},
		AE_Effect_Support_URL {
			DMS_SUPPORT_URL
		}
	}
};
