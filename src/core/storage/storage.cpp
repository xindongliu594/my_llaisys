#include "storage.hpp"

#include "../runtime/runtime.hpp"

namespace llaisys::core {
Storage::Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host)
    : _memory(memory), _size(size), _api(runtime.api()),
      _device_type(runtime.deviceType()), _device_id(runtime.deviceId()),
      _is_host(is_host) {}

Storage::~Storage() {
    if (_is_host) {
        _api->free_host(_memory);
    } else {
        _api->set_device(_device_id);
        _api->free_device(_memory);
    }
}

std::byte *Storage::memory() const {
    return _memory;
}

size_t Storage::size() const {
    return _size;
}

llaisysDeviceType_t Storage::deviceType() const {
    if (isHost()) {
        return LLAISYS_DEVICE_CPU;
    } else {
        return _device_type;
    }
}

int Storage::deviceId() const {
    if (isHost()) {
        return 0;
    } else {
        return _device_id;
    }
}

bool Storage::isHost() const {
    return _is_host;
}
} // namespace llaisys::core
