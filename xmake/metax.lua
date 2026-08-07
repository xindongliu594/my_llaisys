toolchain("mxcc")
    set_kind("standalone")
    set_toolset("cc", "gcc@/opt/maca/mxgpu_llvm/bin/mxcc")
    set_toolset("cxx", "g++@/opt/maca/mxgpu_llvm/bin/mxcc")
    set_toolset("ld", "g++@/opt/maca/mxgpu_llvm/bin/mxcc")
    set_toolset("sh", "g++@/opt/maca/mxgpu_llvm/bin/mxcc")
    set_toolset("ar", "ar@/usr/bin/ar")
toolchain_end()

local function configure_maca_target()
    set_kind("static")
    set_toolchains("mxcc")
    set_languages("cxx17")
    set_warnings("all")
    add_cxflags("-x", "maca", "-offload-arch", "native",
                "--maca-path=/opt/maca", "-fPIC", {force = true})
    add_includedirs("../src/device/metax/compat", {public = true})
    add_includedirs("/opt/maca/include/common", "/opt/maca/include/mcr",
                    "/opt/maca/include/mcblas", "/opt/maca/include")
    add_linkdirs("/opt/maca/lib", {public = true})
    add_links("mcblas", "mcruntime", {public = true})
end

target("llaisys-device-nvidia")
    configure_maca_target()
    add_files("../src/device/metax/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    configure_maca_target()
    add_deps("llaisys-tensor")
    add_files("../src/ops/*/metax/*.cpp")

    on_install(function (target) end)
target_end()
