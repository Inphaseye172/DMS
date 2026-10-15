#pragma once
#include "AE_Effect.h"
template <typename SUITETYPE>
class AEFX_SuiteScoper {
public:
    AEFX_SuiteScoper(const PF_InData* in_data, const char* suite_name, int32_t suite_version, PF_OutData* out_dataP = nullptr,
                     const char* err_stringZ = "Not able to acquire AEFX Suite.")
        : in_(in_data) {
        (void)suite_name; (void)suite_version; (void)out_dataP; (void)err_stringZ;
        if (!in_) throw static_cast<A_Err>(PF_Err_BAD_CALLBACK_PARAM);
        suite_ = nullptr;
    }
    ~AEFX_SuiteScoper() = default;
    const SUITETYPE* operator->() const { return suite_; }
    SUITETYPE* get() const { return suite_; }
private:
    const PF_InData* in_;
    SUITETYPE* suite_;
};
