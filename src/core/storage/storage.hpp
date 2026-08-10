#pragma once
#include "llaisys.h"
#include "llaisys/runtime.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {
class Storage {
private:
    std::byte *_memory;
    size_t _size;
    const LlaisysRuntimeAPI *_api;
    llaisysDeviceType_t _device_type;
    int _device_id;
    bool _is_host;
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

}; // namespace llaisys::core
