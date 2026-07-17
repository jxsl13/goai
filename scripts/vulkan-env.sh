# vulkan-env.sh — source before `-tags vulkan` builds/tests on the linux-amd64 worker.
#
# The immutable distro (bazzite) ships the Vulkan LOADER (libvulkan.so.1) and NVIDIA ICD
# system-wide but no Vulkan SDK headers/pkg-config. This provides both from a header-only
# local install (Vulkan-Headers + a hand-written vulkan.pc pointing at the system loader),
# mirroring what scripts/cuda-pip-env.sh does for the CUDA pip wheels. glslc comes from
# linuxbrew (shaderc). Setup (once):
#
#   mkdir -p ~/.local/share/goai-vulkan-sdk && cd ~/.local/share/goai-vulkan-sdk
#   curl -sL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.280.tar.gz | tar xz
#   mkdir -p lib pkgconfig && ln -sf /usr/lib64/libvulkan.so.1 lib/libvulkan.so
#   cat > pkgconfig/vulkan.pc <<EOF
#   prefix=$HOME/.local/share/goai-vulkan-sdk/Vulkan-Headers-1.3.280/include
#   Name: Vulkan
#   Description: Vulkan headers (local, header-only) + system loader
#   Version: 1.3.280
#   Cflags: -I$HOME/.local/share/goai-vulkan-sdk/Vulkan-Headers-1.3.280/include
#   Libs: -lvulkan
#   EOF
#
# Device pick: vk_bridge.c ranks discrete > integrated, so the RTX 3060 wins over the
# Renoir iGPU automatically. Verified: `go test -tags vulkan ./backend/vulkan/` green on
# this host (TestVulkanCrossReference etc., 2026-07-18).

export PKG_CONFIG_PATH="$HOME/.local/share/goai-vulkan-sdk/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
echo "vulkan-env: PKG_CONFIG_PATH set for $HOME/.local/share/goai-vulkan-sdk" >&2
