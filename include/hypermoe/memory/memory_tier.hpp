#pragma once

#include <string_view>

namespace hypermoe {

enum class MemoryTier {
    Vram,
    Ram,
    PinnedRam,
    Nvme,
};

[[nodiscard]] constexpr std::string_view toString(MemoryTier tier) noexcept {
    switch (tier) {
    case MemoryTier::Vram:
        return "VRAM";
    case MemoryTier::Ram:
        return "RAM";
    case MemoryTier::PinnedRam:
        return "PINNED_RAM";
    case MemoryTier::Nvme:
        return "NVME";
    }
    return "UNKNOWN";
}

} // namespace hypermoe
