# AE SDK compile-check mock

**Not the After Effects SDK.** A minimal set of declarations mirroring the subset of the Adobe After
Effects SDK used by `plugin/ae/`, written from the SDK documentation so the plugin glue can be
*compiled* (never linked or run) on Linux CI. It catches C++ errors in our code (typos, type
mismatches, missing includes, lambda/RAII mistakes). It cannot prove that the real SDK signatures
match; the real build against `AE_SDK_ROOT` on Windows is authoritative.
