// RAII wrappers for host resources. Every acquire has a destructor-driven release, so all exit
// paths (including exceptions) free PF_Handles, check params back in, and check in layer pixels.
#pragma once

#include "ae_includes.h"

#include <cstring>
#include <utility>

namespace dms_ae {

// Owns a PF_Handle allocated through PF_HandleSuite1. release() hands ownership to the host.
class HostHandle {
public:
    HostHandle(const PF_InData* in_data, PF_OutData* out_data) : suite_(in_data, DMS_HANDLE_SUITE, out_data) {}
    ~HostHandle() { reset(); }
    HostHandle(const HostHandle&) = delete;
    HostHandle& operator=(const HostHandle&) = delete;

    bool allocate(A_HandleSize size) {
        reset();
        h_ = suite_->host_new_handle(size);
        return h_ != nullptr;
    }
    PF_Handle get() const noexcept { return h_; }
    PF_Handle release() noexcept { return std::exchange(h_, nullptr); }
    void reset() noexcept {
        if (h_) {
            suite_->host_dispose_handle(h_);
            h_ = nullptr;
        }
    }

private:
    AEFX_SuiteScoper<DmsHandleSuite> suite_;
    PF_Handle h_ = nullptr;
};

// Locks a PF_Handle for the scope.
class HandleLock {
public:
    HandleLock(const PF_InData* in_data, PF_Handle h) : suite_(in_data, DMS_HANDLE_SUITE), h_(h) {
        if (h_) {
            p_ = suite_->host_lock_handle(h_);
            size_ = suite_->host_get_handle_size(h_);
        }
    }
    ~HandleLock() {
        if (h_ && p_) suite_->host_unlock_handle(h_);
    }
    HandleLock(const HandleLock&) = delete;
    HandleLock& operator=(const HandleLock&) = delete;
    void* data() const noexcept { return p_; }
    size_t size() const noexcept { return static_cast<size_t>(size_); }

private:
    AEFX_SuiteScoper<DmsHandleSuite> suite_;
    PF_Handle h_;
    void* p_ = nullptr;
    A_HandleSize size_ = 0;
};

// PF_CHECKOUT_PARAM / PF_CHECKIN_PARAM pair (used where params[] is not valid: pre-render, dynamic flags).
class ParamCheckout {
public:
    ParamCheckout(PF_InData* in_data, PF_ParamIndex index) : in_(in_data) {
        AEFX_CLR_STRUCT(def_);
        err_ = PF_CHECKOUT_PARAM(in_data, index, in_data->current_time, in_data->time_step, in_data->time_scale, &def_);
        ok_ = err_ == PF_Err_NONE;
    }
    ~ParamCheckout() {
        if (ok_) PF_CHECKIN_PARAM(in_, &def_);
    }
    ParamCheckout(const ParamCheckout&) = delete;
    ParamCheckout& operator=(const ParamCheckout&) = delete;
    bool ok() const noexcept { return ok_; }
    PF_Err err() const noexcept { return err_; }
    const PF_ParamDef& def() const noexcept { return def_; }

private:
    PF_InData* in_;
    PF_ParamDef def_;
    PF_Err err_ = PF_Err_NONE;
    bool ok_ = false;
};

// Checks in SmartFX layer pixels on scope exit (even on error paths).
class SmartLayerCheckout {
public:
    SmartLayerCheckout(PF_InData* in_data, PF_SmartRenderExtra* extra, A_long checkout_id) : in_(in_data), extra_(extra), id_(checkout_id) {
        err_ = extra_->cb->checkout_layer_pixels(in_->effect_ref, id_, &world_);
        if (err_ != PF_Err_NONE) world_ = nullptr;
    }
    ~SmartLayerCheckout() {
        if (err_ == PF_Err_NONE && extra_->cb->checkin_layer_pixels) extra_->cb->checkin_layer_pixels(in_->effect_ref, id_);
    }
    SmartLayerCheckout(const SmartLayerCheckout&) = delete;
    SmartLayerCheckout& operator=(const SmartLayerCheckout&) = delete;
    PF_EffectWorld* world() const noexcept { return world_; }
    PF_Err err() const noexcept { return err_; }

private:
    PF_InData* in_;
    PF_SmartRenderExtra* extra_;
    A_long id_;
    PF_EffectWorld* world_ = nullptr;
    PF_Err err_ = PF_Err_NONE;
};

// AEGP effect / stream references (AE only; never used in Premiere).
class AegpEffectRef {
public:
    AegpEffectRef(PF_InData* in_data, AEGP_PluginID id)
        : pfi_(in_data, DMS_PFINTERFACE_SUITE), effects_(in_data, DMS_EFFECT_SUITE) {
        if (pfi_->AEGP_GetNewEffectForEffect(id, in_data->effect_ref, &h_) != A_Err_NONE) h_ = nullptr;
    }
    ~AegpEffectRef() {
        if (h_) effects_->AEGP_DisposeEffect(h_);
    }
    AegpEffectRef(const AegpEffectRef&) = delete;
    AegpEffectRef& operator=(const AegpEffectRef&) = delete;
    AEGP_EffectRefH get() const noexcept { return h_; }

private:
    AEFX_SuiteScoper<DmsPFInterfaceSuite> pfi_;
    AEFX_SuiteScoper<DmsEffectSuite> effects_;
    AEGP_EffectRefH h_ = nullptr;
};

class AegpStreamRef {
public:
    AegpStreamRef(PF_InData* in_data, AEGP_PluginID id, AEGP_EffectRefH effect, PF_ParamIndex index) : streams_(in_data, DMS_STREAM_SUITE) {
        if (!effect || streams_->AEGP_GetNewEffectStreamByIndex(id, effect, index, &h_) != A_Err_NONE) h_ = nullptr;
    }
    ~AegpStreamRef() {
        if (h_) streams_->AEGP_DisposeStream(h_);
    }
    AegpStreamRef(const AegpStreamRef&) = delete;
    AegpStreamRef& operator=(const AegpStreamRef&) = delete;
    AEGP_StreamRefH get() const noexcept { return h_; }

private:
    AEFX_SuiteScoper<DmsStreamSuite> streams_;
    AEGP_StreamRefH h_ = nullptr;
};

}  // namespace dms_ae
