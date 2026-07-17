//go:build vulkan && cgo

// Vulkan compute side of the vulkan backend (§T43). Compiled only under
// `-tags vulkan` with the Vulkan loader. A process-wide instance/device/queue is
// created lazily; per-shader pipeline objects (module, layouts, pipeline,
// descriptor set) are created once and cached across calls (§T95/§T335,
// vkCreateComputePipelines dominates the first call). Storage buffers come from
// a process-wide size-classed pool and the command buffer is a single reused
// one (§T335) — per call only the host↔device copies and the dispatch itself
// remain. Confirmed vs the Vulkan spec + compute tutorials (§R44).

#include "vk_bridge.h"
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

static VkInstance       gInstance = VK_NULL_HANDLE;
static VkPhysicalDevice gPhys     = VK_NULL_HANDLE;
static VkDevice         gDevice   = VK_NULL_HANDLE;
static VkQueue          gQueue    = VK_NULL_HANDLE;
static uint32_t         gQueueFamily = 0;
static int              gInitTried = 0;
static int              gInitOK    = 0;
static int              gAtomicFloat = 0; // VK_EXT_shader_atomic_float enabled? (§T90)
static int              gCoopMat = 0;     // VK_KHR_cooperative_matrix enabled? (Tw-COOPMAT)
typedef struct { VkBuffer buf; VkDeviceMemory mem; } ResidentBuf;

// find_compute_queue returns the index of a queue family with COMPUTE, or -1.
static int find_compute_queue(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    if (n == 0) return -1;
    VkQueueFamilyProperties* props = malloc(n * sizeof(*props));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, props);
    int found = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { found = (int)i; break; }
    }
    free(props);
    return found;
}

// has_instance_ext reports whether the loader exposes an instance extension.
static int has_instance_ext(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    if (n == 0) return 0;
    VkExtensionProperties* ex = malloc(n * sizeof(*ex));
    vkEnumerateInstanceExtensionProperties(NULL, &n, ex);
    int found = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (strcmp(ex[i].extensionName, name) == 0) { found = 1; break; }
    }
    free(ex);
    return found;
}

// has_device_ext reports whether a physical device exposes a device extension.
static int has_device_ext(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    if (n == 0) return 0;
    VkExtensionProperties* ex = malloc(n * sizeof(*ex));
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, ex);
    int found = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (strcmp(ex[i].extensionName, name) == 0) { found = 1; break; }
    }
    free(ex);
    return found;
}

// device_score ranks a physical device for compute: discrete GPU > integrated > virtual/CPU/other.
// -1 means unusable (no compute queue). Used to prefer the fastest GPU on a multi-GPU host (e.g. an
// NVIDIA dGPU over an AMD iGPU) — §V4.
static int device_score(VkPhysicalDevice pd) {
    if (find_compute_queue(pd) < 0) return -1;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 1;
        default:                                     return 0; // CPU / other
    }
}

// attempt_device creates the logical device on `pd`; on success sets gPhys/gQueueFamily/gDevice/gQueue
// (+gAtomicFloat) and returns 0, else -1. Extracted so ensure_init can try candidates best-first and
// fall back if a driver rejects device creation.
static int attempt_device(VkPhysicalDevice pd) {
    int qf = find_compute_queue(pd);
    if (qf < 0) return -1;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = (uint32_t)qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // Portability (MoltenVK) needs VK_KHR_portability_subset at device creation; absent elsewhere.
    const char* devExts[3];
    uint32_t nDevExts = 0;
    if (has_device_ext(pd, "VK_KHR_portability_subset")) {
        devExts[nDevExts++] = "VK_KHR_portability_subset";
    }
    // VK_EXT_shader_atomic_float: float atomicAdd for the MHA-backward dK/dV accumulation (§T90).
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFeat = {0};
    atomicFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    void* pNext = NULL;
    int atomic = 0;
    if (has_device_ext(pd, "VK_EXT_shader_atomic_float")) {
        devExts[nDevExts++] = "VK_EXT_shader_atomic_float";
        atomicFeat.shaderBufferFloat32Atomics = VK_TRUE;
        atomicFeat.shaderBufferFloat32AtomicAdd = VK_TRUE;
        pNext = &atomicFeat;
        atomic = 1;
    }
    // VK_KHR_cooperative_matrix (Tw-COOPMAT): tensor-core GEMM from GLSL. Needs the
    // core-1.2 shaderFloat16 + storageBuffer16BitAccess + Vulkan memory model features;
    // gated on device support so MoltenVK/older stacks keep the exact previous path.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat = {0};
    VkPhysicalDeviceVulkan11Features v11 = {0};
    VkPhysicalDeviceVulkan12Features v12 = {0};
    int coop = 0;
    VkPhysicalDeviceProperties pdProps;
    vkGetPhysicalDeviceProperties(pd, &pdProps);
    if (pdProps.apiVersion >= VK_API_VERSION_1_2 && has_device_ext(pd, "VK_KHR_cooperative_matrix")) {
        devExts[nDevExts++] = "VK_KHR_cooperative_matrix";
        coopFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopFeat.cooperativeMatrix = VK_TRUE;
        v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        v11.storageBuffer16BitAccess = VK_TRUE;
        v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.shaderFloat16 = VK_TRUE;
        v12.vulkanMemoryModel = VK_TRUE;
        v12.vulkanMemoryModelDeviceScope = VK_TRUE;
        coopFeat.pNext = pNext;
        v11.pNext = &coopFeat;
        v12.pNext = &v11;
        pNext = &v12;
        coop = 1;
    }

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = pNext;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nDevExts;
    dci.ppEnabledExtensionNames = nDevExts ? devExts : NULL;
    VkDevice dev = NULL;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) return -1;

    gPhys = pd;
    gQueueFamily = (uint32_t)qf;
    gDevice = dev;
    gAtomicFloat = atomic;
    gCoopMat = coop;
    vkGetDeviceQueue(gDevice, gQueueFamily, 0, &gQueue);
    return 0;
}

static int ensure_init(void) {
    if (gInitTried) return gInitOK ? 0 : -1;
    gInitTried = 1;

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "goai";
    // Request up to Vulkan 1.3 when the loader supports it (VK_KHR_cooperative_matrix's
    // GLSL side needs the core-1.2 Vulkan memory model); older loaders (MoltenVK) keep 1.1.
    uint32_t instVer = VK_API_VERSION_1_1;
    {
        uint32_t v = 0;
        if (vkEnumerateInstanceVersion(&v) == VK_SUCCESS && v >= VK_API_VERSION_1_3) instVer = VK_API_VERSION_1_3;
        else if (v >= VK_API_VERSION_1_2) instVer = VK_API_VERSION_1_2;
    }
    app.apiVersion = instVer;

    // Portability drivers (MoltenVK on macOS) are hidden unless the instance
    // opts in via VK_KHR_portability_enumeration + the ENUMERATE_PORTABILITY
    // flag. On native Linux/Windows the extension is absent → we skip it and
    // enumerate normally. One code path, portable across all hosts.
    const char* instExts[1];
    uint32_t nInstExts = 0;
    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (has_instance_ext("VK_KHR_portability_enumeration")) {
        instExts[nInstExts++] = "VK_KHR_portability_enumeration";
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    ici.enabledExtensionCount = nInstExts;
    ici.ppEnabledExtensionNames = nInstExts ? instExts : NULL;
    if (vkCreateInstance(&ici, NULL, &gInstance) != VK_SUCCESS) return -1;

    uint32_t nd = 0;
    if (vkEnumeratePhysicalDevices(gInstance, &nd, NULL) != VK_SUCCESS || nd == 0) return -1;
    VkPhysicalDevice* devs = malloc(nd * sizeof(*devs));
    vkEnumeratePhysicalDevices(gInstance, &nd, devs);

    // Try compute-capable devices BEST-FIRST (discrete GPU > integrated > virtual > CPU) and use the
    // first whose logical device creates. On a multi-GPU host this picks the fastest GPU (e.g. an
    // NVIDIA RTX 3060 over an AMD Renoir iGPU) instead of whatever enumerates first, and it falls back
    // if a driver rejects device creation — previously a create failure on device 0 aborted init and
    // left the machine reporting "no compute device" despite a working GPU. §V4.
    int ok = -1;
    for (int rank = 3; rank >= 0 && ok < 0; --rank) {
        for (uint32_t i = 0; i < nd; ++i) {
            if (device_score(devs[i]) != rank) continue;
            if (attempt_device(devs[i]) == 0) { ok = 0; break; }
        }
    }
    free(devs);
    if (ok < 0) return -1;
    gInitOK = 1;
    return 0;
}

int vk_available(void) { return ensure_init() == 0 ? 1 : 0; }

// vk_atomic_float reports whether the device enabled VK_EXT_shader_atomic_float
// (required by the MHA-backward kernel's shared dK/dV accumulation, §T90).
int vk_atomic_float(void) { return ensure_init() == 0 && gAtomicFloat ? 1 : 0; }

// pick host-visible + coherent memory type in `bits`, or -1.
static int mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gPhys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return (int)i;
        }
    }
    return -1;
}

// make_buffer creates a host-visible|coherent storage buffer of `size` bytes.
static int make_buffer(VkDeviceSize size, VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(gDevice, &bci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gDevice, *buf, &req);
    int mt = mem_type(req.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt < 0) return -1;

    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = (uint32_t)mt;
    if (vkAllocateMemory(gDevice, &mai, NULL, mem) != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(gDevice, *buf, *mem, 0) != VK_SUCCESS) return -1;
    return 0;
}

static void upload(VkDeviceMemory mem, const void* src, size_t n) {
    void* p = NULL;
    vkMapMemory(gDevice, mem, 0, n, 0, &p);
    memcpy(p, src, n);
    vkUnmapMemory(gDevice, mem);
}

static void download(VkDeviceMemory mem, void* dst, size_t n) {
    void* p = NULL;
    vkMapMemory(gDevice, mem, 0, n, 0, &p);
    memcpy(dst, p, n);
    vkUnmapMemory(gDevice, mem);
}

// ---- process-wide buffer pool + reused command buffer (§T335) --------------
// vkCreateBuffer+vkAllocateMemory(+destroy/free) per storage buffer per call was
// the dominant per-call cost after the §T95 pipeline cache (vkAllocateMemory is
// a full driver allocation; MoltenVK backs it with a fresh MTLBuffer). Buffers
// are now pooled process-wide: capacities are rounded up to a power of two and
// reused best-fit across calls (training/inference loops repeat the same shapes,
// so the pool converges after the first step). Reused buffers may hold stale
// bytes — safe because every kernel either writes each output element it
// downloads or reads an accumulator that is uploaded pre-zeroed (up=1).
// The backend is synchronous (single queue, waitIdle per call); gLock serializes
// dispatches so the pool, the per-pipeline descriptor sets, and the single
// reused command buffer are safe under concurrent Execute callers.
static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize cap; int inUse; } PoolBuf;
#define VK_POOL_MAX 128
static PoolBuf      gPool[VK_POOL_MAX];
static int          gNPool = 0;
static VkDeviceSize gPoolBytes = 0;
// Beyond this the pool stops growing and oversize requests fall back to a
// transient per-call buffer (old behavior) — bounds device memory held idle.
static const VkDeviceSize kPoolByteCap = (VkDeviceSize)512 << 20;

static VkDeviceSize pool_round(VkDeviceSize n) {
    VkDeviceSize c = 256;
    while (c < n) c <<= 1;
    return c;
}

// pool_acquire hands out an idle pooled buffer with capacity >= size (best
// fit), growing the pool on miss. Returns the pool index, or -1 for a transient
// unpooled buffer (pool table/byte cap reached; caller destroys it), or -2 on
// allocation failure.
static int pool_acquire(VkDeviceSize size, VkBuffer* buf, VkDeviceMemory* mem) {
    int best = -1;
    for (int i = 0; i < gNPool; ++i) {
        if (gPool[i].inUse || gPool[i].cap < size) continue;
        if (best < 0 || gPool[i].cap < gPool[best].cap) best = i;
    }
    if (best >= 0) {
        gPool[best].inUse = 1;
        *buf = gPool[best].buf;
        *mem = gPool[best].mem;
        return best;
    }
    VkDeviceSize cap = pool_round(size);
    int pooled = gNPool < VK_POOL_MAX && gPoolBytes + cap <= kPoolByteCap;
    if (make_buffer(pooled ? cap : size, buf, mem) != 0) {
        if (*buf) vkDestroyBuffer(gDevice, *buf, NULL);
        if (*mem) vkFreeMemory(gDevice, *mem, NULL);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        return -2;
    }
    if (!pooled) return -1;
    gPool[gNPool] = (PoolBuf){ *buf, *mem, cap, 1 };
    gPoolBytes += cap;
    return gNPool++;
}

static void pool_release(int idx, VkBuffer buf, VkDeviceMemory mem) {
    if (idx >= 0) { gPool[idx].inUse = 0; return; }
    if (buf) vkDestroyBuffer(gDevice, buf, NULL);
    if (mem) vkFreeMemory(gDevice, mem, NULL);
}

// One command pool + one command buffer for the process, reset per call —
// replaces a vkCreateCommandPool/vkAllocateCommandBuffers/destroy round trip
// per dispatch. Valid because dispatches are serialized (gLock) and each call
// waits the queue idle before returning.
static VkCommandPool   gCmdPool = VK_NULL_HANDLE;
static VkCommandBuffer gCmd     = VK_NULL_HANDLE;

static int ensure_cmd(void) {
    if (gCmd) return 0;
    VkCommandPoolCreateInfo cpc = {0};
    cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpc.queueFamilyIndex = gQueueFamily;
    if (vkCreateCommandPool(gDevice, &cpc, NULL, &gCmdPool) != VK_SUCCESS) return -1;
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = gCmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(gDevice, &cbai, &gCmd) != VK_SUCCESS) {
        vkDestroyCommandPool(gDevice, gCmdPool, NULL);
        gCmdPool = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

// One descriptor pool for the process; each cached pipeline allocates its
// descriptor set from it once (§T335) and rewrites the buffer bindings per call
// (legal: the set is never in flight when updated — waitIdle per dispatch).
static VkDescriptorPool gDescPool = VK_NULL_HANDLE;

// One cached compute pipeline per distinct shader (keyed by the embedded .spv pointer). Sized
// above the number of compute shaders the backend ships (matmul, mha, flashattn, mha_bwd,
// conv2d, conv2d_bwd, qmatmul_q8/q4k/q6k, …) — overflowing it makes pipeline_for fail with -4
// only once many shaders have run in one process, so keep headroom as kernels are added.
#define VK_MAX_PIPELINES 48

static int ensure_desc_pool(void) {
    if (gDescPool) return 0;
    VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * VK_MAX_PIPELINES };
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = VK_MAX_PIPELINES;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    if (vkCreateDescriptorPool(gDevice, &dpci, NULL, &gDescPool) != VK_SUCCESS) return -1;
    return 0;
}

// ---- per-shader pipeline cache (§T95) -------------------------------------
// vkCreateComputePipelines compiles the SPIR-V (→Metal on MoltenVK) and dominates
// the per-call cost at small/medium sizes. The descriptor-set layout, pipeline
// layout, shader module, and pipeline are FIXED per shader, so they are created
// once and cached, keyed by the embedded SPIR-V pointer (stable for the process).
// Only the data-dependent buffers, descriptor set, and command buffer stay
// per-call. The backend is synchronous (one queue, waitIdle per call), so no lock
// is needed on the small cache.
typedef struct {
    const uint32_t*       spv;
    VkShaderModule        module;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      plLayout;
    VkPipeline            pipe;
    VkDescriptorSet       dset; // allocated once from gDescPool, rewritten per call (§T335)
} PipeCache;

static PipeCache gPipes[VK_MAX_PIPELINES];
static int gNPipes = 0;

// pipeline_for returns the cached pipeline objects for `spv` (nbuf storage-buffer
// bindings + pushLen bytes of push constants), creating and caching them on first
// use. Returns 0 on success (*out set), nonzero on failure.
static int pipeline_for(const uint32_t* spv, int spvLen, int nbuf, uint32_t pushLen, PipeCache** out) {
    for (int i = 0; i < gNPipes; ++i) {
        if (gPipes[i].spv == spv) { *out = &gPipes[i]; return 0; }
    }
    if (gNPipes >= VK_MAX_PIPELINES) return -4;
    PipeCache pc = {0};
    pc.spv = spv;

    VkDescriptorSetLayoutBinding binds[8];
    for (int i = 0; i < nbuf; ++i) {
        binds[i] = (VkDescriptorSetLayoutBinding){0};
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (uint32_t)nbuf;
    dslci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(gDevice, &dslci, NULL, &pc.dsl) != VK_SUCCESS) return -3;

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = (size_t)spvLen;
    smci.pCode = spv;
    if (vkCreateShaderModule(gDevice, &smci, NULL, &pc.module) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(gDevice, pc.dsl, NULL);
        return -4;
    }

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushLen };
    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pc.dsl;
    plci.pushConstantRangeCount = pushLen ? 1 : 0;
    plci.pPushConstantRanges = pushLen ? &pcr : NULL;
    if (vkCreatePipelineLayout(gDevice, &plci, NULL, &pc.plLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(gDevice, pc.module, NULL);
        vkDestroyDescriptorSetLayout(gDevice, pc.dsl, NULL);
        return -4;
    }

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = pc.plLayout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = pc.module;
    cpci.stage.pName = "main";
    if (vkCreateComputePipelines(gDevice, VK_NULL_HANDLE, 1, &cpci, NULL, &pc.pipe) != VK_SUCCESS) {
        vkDestroyPipelineLayout(gDevice, pc.plLayout, NULL);
        vkDestroyShaderModule(gDevice, pc.module, NULL);
        vkDestroyDescriptorSetLayout(gDevice, pc.dsl, NULL);
        return -4;
    }

    // The descriptor set lives as long as the pipeline: allocated once here,
    // its buffer bindings rewritten per dispatch (§T335).
    int descOK = ensure_desc_pool() == 0;
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = gDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pc.dsl;
    if (!descOK ||
        vkAllocateDescriptorSets(gDevice, &dsai, &pc.dset) != VK_SUCCESS) {
        vkDestroyPipeline(gDevice, pc.pipe, NULL);
        vkDestroyPipelineLayout(gDevice, pc.plLayout, NULL);
        vkDestroyShaderModule(gDevice, pc.module, NULL);
        vkDestroyDescriptorSetLayout(gDevice, pc.dsl, NULL);
        return -3;
    }

    gPipes[gNPipes] = pc;
    *out = &gPipes[gNPipes];
    gNPipes++;
    return 0;
}

// dset_write rewrites a cached descriptor set's storage-buffer bindings (legal
// whenever the set is not in flight — every dispatch waits the queue idle).
static void dset_write(VkDescriptorSet dset, int nbuf, const VkBuffer* buf, const VkDeviceSize* lens) {
    VkDescriptorBufferInfo bufInfo[8];
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < nbuf; ++i) {
        bufInfo[i] = (VkDescriptorBufferInfo){ buf[i], 0, lens[i] };
        writes[i] = (VkWriteDescriptorSet){0};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfo[i];
    }
    vkUpdateDescriptorSets(gDevice, (uint32_t)nbuf, writes, 0, NULL);
}

// vk_dispatch is the generic launcher shared by every op (§T95): nbuf host-visible
// storage buffers at bindings 0..nbuf-1 (up[i]!=0 → staged from data[i] first;
// down[i]!=0 → read back after), optional push constants, and a 3-D workgroup
// count. Pipeline objects and the descriptor set come from the per-shader cache,
// buffers from the process-wide pool, and the command buffer is the reused one —
// per call only uploads, descriptor rewrites, the dispatch, and downloads (§T335).
// vk_dispatch_pre is vk_dispatch with optional PRE-MADE buffers: for a slot i where preBuf[i] is
// non-null, that buffer is used as-is (device-resident, §T156) — not acquired, uploaded, or
// released here. preBuf may be NULL (all buffers pooled, the normal path).
static int vk_dispatch_locked(const uint32_t* spv, int spvLen,
                       int nbuf, const VkDeviceSize* lens, void* const* data,
                       const int* up, const int* down,
                       const void* push, uint32_t pushLen,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const VkBuffer* preBuf) {
    if (ensure_init() != 0) return -1;
    if (nbuf < 1 || nbuf > 8) return -2;

    PipeCache* pc = NULL;
    if (pipeline_for(spv, spvLen, nbuf, pushLen, &pc) != 0) return -4;

    VkBuffer buf[8];
    VkDeviceMemory mem[8];
    int poolIdx[8]; // pool slot per buffer; -1 transient, -3 pre-made/not acquired
    for (int i = 0; i < nbuf; ++i) { buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE; poolIdx[i] = -3; }
    int rc = -2;

    for (int i = 0; i < nbuf; ++i) {
        if (preBuf && preBuf[i]) { buf[i] = preBuf[i]; continue; } // resident buffer: use as-is
        poolIdx[i] = pool_acquire(lens[i], &buf[i], &mem[i]);
        if (poolIdx[i] == -2) { poolIdx[i] = -3; goto done; }
        if (up && up[i]) upload(mem[i], data[i], (size_t)lens[i]);
    }

    dset_write(pc->dset, nbuf, buf, lens);

    if (ensure_cmd() != 0) { rc = -5; goto done; }
    vkResetCommandBuffer(gCmd, 0);

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(gCmd, &cbbi);
    vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->pipe);
    vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->plLayout, 0, 1, &pc->dset, 0, NULL);
    if (pushLen) vkCmdPushConstants(gCmd, pc->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushLen, push);
    vkCmdDispatch(gCmd, gx, gy, gz);
    vkEndCommandBuffer(gCmd);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gCmd;
    if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) { rc = -6; goto done; }
    if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) { rc = -6; goto done; }

    for (int i = 0; i < nbuf; ++i) {
        if (down && down[i]) download(mem[i], data[i], (size_t)lens[i]);
    }
    rc = 0;

done:
    for (int i = 0; i < nbuf; ++i) {
        if (poolIdx[i] == -3) continue; // resident buffer or never acquired: not owned here
        pool_release(poolIdx[i], buf[i], mem[i]);
    }
    return rc;
}

static int vk_dispatch_pre(const uint32_t* spv, int spvLen,
                       int nbuf, const VkDeviceSize* lens, void* const* data,
                       const int* up, const int* down,
                       const void* push, uint32_t pushLen,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const VkBuffer* preBuf) {
    pthread_mutex_lock(&gLock);
    int rc = vk_dispatch_locked(spv, spvLen, nbuf, lens, data, up, down,
                                push, pushLen, gx, gy, gz, preBuf);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_dispatch is the all-per-call launcher used by every op (§T95): no pre-made buffers.
static int vk_dispatch(const uint32_t* spv, int spvLen,
                       int nbuf, const VkDeviceSize* lens, void* const* data,
                       const int* up, const int* down,
                       const void* push, uint32_t pushLen,
                       uint32_t gx, uint32_t gy, uint32_t gz) {
    return vk_dispatch_pre(spv, spvLen, nbuf, lens, data, up, down, push, pushLen, gx, gy, gz, NULL);
}

// ---- op wrappers (build buffer arrays + push block, then vk_dispatch) ------

// vk_coopmat reports whether the active device enabled VK_KHR_cooperative_matrix.
int vk_coopmat(void) { return ensure_init() == 0 && gCoopMat ? 1 : 0; }

// vk_coopmat_gemm_f16: Tw-COOPMAT probe — C[M,N]f32 = A[M,K]f16 · B[K,N]f16 on the
// cooperative-matrix (tensor-core) pipeline. Host must guarantee M%16==0, N%64==0,
// K%16==0 and vk_coopmat()==1. Each workgroup = one subgroup computing a 16×64 C tile.
int vk_coopmat_gemm_f16(const uint32_t* spv, int spvLen,
                        const void* Ah, const void* Bh, float* C,
                        int M, int K, int N) {
    if (!gCoopMat) return -9;
    VkDeviceSize lens[3] = {
        (VkDeviceSize)M * K * 2,
        (VkDeviceSize)K * N * 2,
        (VkDeviceSize)M * N * sizeof(float),
    };
    void* data[3] = { (void*)Ah, (void*)Bh, C };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    int32_t pc[3] = { M, K, N };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, pc, sizeof(pc),
                       (uint32_t)N / 64u, (uint32_t)M / 16u, 1u);
}

// vk_coopmat_gemm_f16_res: the resident-buffer twin of vk_coopmat_gemm_f16 — A/B/C live in
// pre-uploaded device buffers (vk_devbuf_upload handles), so a timed loop measures the KERNEL
// rate, not the ~25 MB/call host round-trip the pooled path pays. C stays on device; read it
// back with vk_devbuf_download.
int vk_coopmat_gemm_f16_res(const uint32_t* spv, int spvLen,
                            void* ah, void* bh, void* ch,
                            int M, int K, int N) {
    if (!gCoopMat) return -9;
    ResidentBuf* ra = (ResidentBuf*)ah;
    ResidentBuf* rb = (ResidentBuf*)bh;
    ResidentBuf* rc = (ResidentBuf*)ch;
    VkDeviceSize lens[3] = {
        (VkDeviceSize)M * K * 2,
        (VkDeviceSize)K * N * 2,
        (VkDeviceSize)M * N * sizeof(float),
    };
    void* data[3] = { NULL, NULL, NULL };
    int up[3] = {0, 0, 0}, down[3] = {0, 0, 0};
    int32_t pc[3] = { M, K, N };
    VkBuffer preBuf[3] = { ra->buf, rb->buf, rc->buf };
    return vk_dispatch_pre(spv, spvLen, 3, lens, data, up, down, pc, sizeof(pc),
                           (uint32_t)N / 64u, (uint32_t)M / 16u, 1u, preBuf);
}

int vk_matmul_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* C,
                  int M, int K, int N, int transA, int transB) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)M * K * sizeof(float), // A holds M·K elements in either layout
        (VkDeviceSize)K * N * sizeof(float),
        (VkDeviceSize)M * N * sizeof(float),
    };
    void* data[3] = { (void*)A, (void*)B, C };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    // {M,K,N,transA,transB} — a transposed operand is read transposed in the shader
    // (§T356) instead of materialized with a CPU strided-gather copy.
    int32_t pc[6] = { M, K, N, transA, transB, 0 }; // acc=0: host path always overwrites (SPEC T613)
    // tiled GEMM: each 16×16 workgroup covers a 16×16 output tile (§T94; register
    // blocking gave no gain on MoltenVK, §B39).
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, pc, sizeof(pc),
                       ((uint32_t)N + 15u) / 16u, ((uint32_t)M + 15u) / 16u, 1u);
}

int vk_rmsnorm_f32(const uint32_t* spv, int spvLen,
                   const float* X, const float* G, float* O,
                   int rows, int dim, float eps) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)rows * dim * sizeof(float),
        (VkDeviceSize)dim * sizeof(float),
        (VkDeviceSize)rows * dim * sizeof(float),
    };
    void* data[3] = { (void*)X, (void*)G, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t rows; int32_t dim; float eps; } pc = { rows, dim, eps };
    // cooperative (§T345): ONE 256-invocation workgroup per row.
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)rows, 1u, 1u);
}

int vk_rope_f32(const uint32_t* spv, int spvLen,
                const float* Q, const float* inv, float* O,
                int seq, int width, int heads, int hd, int half, int posOffset, float posDiv) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)seq * width * sizeof(float),
        (VkDeviceSize)half * sizeof(float),
        (VkDeviceSize)seq * width * sizeof(float),
    };
    void* data[3] = { (void*)Q, (void*)inv, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    // off=0: the host per-op path never uses fused-QKV sub-row views (SPEC T613); the
    // struct must still match the shader's widened push block byte-for-byte.
    struct { int32_t seq, width, heads, hd, half, posOffset, off; float posDiv; } pc = {
        seq, width, heads, hd, half, posOffset, 0, posDiv };
    // one invocation per rotated pair: seq·heads·(hd/2) threads, 64 per workgroup.
    uint32_t total = (uint32_t)seq * (uint32_t)heads * (uint32_t)half;
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       (total + 63u) / 64u, 1u, 1u);
}

int vk_softmax_f32(const uint32_t* spv, int spvLen,
                   const float* X, float* O, int rows, int dim) {
    VkDeviceSize lens[2] = {
        (VkDeviceSize)rows * dim * sizeof(float),
        (VkDeviceSize)rows * dim * sizeof(float),
    };
    void* data[2] = { (void*)X, O };
    int up[2] = {1, 0}, down[2] = {0, 1};
    struct { int32_t rows; int32_t dim; } pc = { rows, dim };
    // cooperative (§T346): ONE 256-invocation workgroup per row.
    return vk_dispatch(spv, spvLen, 2, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)rows, 1u, 1u);
}

int vk_layernorm_f32(const uint32_t* spv, int spvLen,
                     const float* X, const float* G, const float* B, float* O,
                     int rows, int dim, float eps) {
    VkDeviceSize lens[4] = {
        (VkDeviceSize)rows * dim * sizeof(float),
        (VkDeviceSize)dim * sizeof(float),
        (VkDeviceSize)dim * sizeof(float),
        (VkDeviceSize)rows * dim * sizeof(float),
    };
    void* data[4] = { (void*)X, (void*)G, (void*)B, O };
    int up[4] = {1, 1, 1, 0}, down[4] = {0, 0, 0, 1};
    struct { int32_t rows; int32_t dim; float eps; } pc = { rows, dim, eps };
    // cooperative (§T346): ONE 256-invocation workgroup per row.
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)rows, 1u, 1u);
}

int vk_embed_backward_f32(const uint32_t* spv, int spvLen,
                          const float* Idx, const float* G, float* DTable, int n, int d, int m) {
    if (vk_atomic_float() != 1) {
        return -7; // no float atomics → caller uses the reference
    }
    VkDeviceSize lens[3] = {
        (VkDeviceSize)m * sizeof(float),     // Idx
        (VkDeviceSize)m * d * sizeof(float), // G (upstream)
        (VkDeviceSize)n * d * sizeof(float), // DTable (atomic accumulator, uploaded zero)
    };
    void* data[3] = { (void*)Idx, (void*)G, DTable };
    int up[3] = {1, 1, 1}, down[3] = {0, 0, 1};
    struct { int32_t n; int32_t d; int32_t m; } pc = { n, d, m };
    // one invocation per (token, dim); 256 per workgroup.
    uint32_t total = (uint32_t)m * (uint32_t)d;
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       (total + 255u) / 256u, 1u, 1u);
}

int vk_crossentropy_backward_f32(const uint32_t* spv, int spvLen,
                                 const float* Z, const float* T, float* DZ,
                                 int b, int c, float gv, float eps, float zl) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)b * c * sizeof(float), // Z
        (VkDeviceSize)b * sizeof(float),     // T (targets as float)
        (VkDeviceSize)b * c * sizeof(float), // DZ (out)
    };
    void* data[3] = { (void*)Z, (void*)T, DZ };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t b; int32_t c; float gv; float eps; float zl; } pc = { b, c, gv, eps, zl };
    // one invocation per row; 64 rows per workgroup.
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)b + 63u) / 64u, 1u, 1u);
}

int vk_rmsnorm_backward_f32(const uint32_t* spv, int spvLen,
                            const float* X, const float* Gamma, const float* G,
                            float* DX, float* DGamma, int rows, int dim, float eps) {
    if (vk_atomic_float() != 1) {
        return -7; // no float atomics → caller uses the reference
    }
    VkDeviceSize lens[5] = {
        (VkDeviceSize)rows * dim * sizeof(float), // X
        (VkDeviceSize)dim * sizeof(float),        // Gamma
        (VkDeviceSize)rows * dim * sizeof(float), // G (upstream)
        (VkDeviceSize)rows * dim * sizeof(float), // DX (out)
        (VkDeviceSize)dim * sizeof(float),        // DGamma (atomic accumulator, uploaded zero)
    };
    void* data[5] = { (void*)X, (void*)Gamma, (void*)G, DX, DGamma };
    int up[5] = {1, 1, 1, 0, 1}, down[5] = {0, 0, 0, 1, 1};
    struct { int32_t rows; int32_t dim; float eps; } pc = { rows, dim, eps };
    // cooperative (§T347): ONE 256-invocation workgroup per row.
    return vk_dispatch(spv, spvLen, 5, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)rows, 1u, 1u);
}

int vk_layernorm_backward_f32(const uint32_t* spv, int spvLen,
                              const float* X, const float* Gamma, const float* G,
                              float* DX, float* DGamma, float* DBeta, int rows, int dim, float eps) {
    if (vk_atomic_float() != 1) {
        return -7; // no float atomics → caller uses the reference
    }
    VkDeviceSize lens[6] = {
        (VkDeviceSize)rows * dim * sizeof(float), // X
        (VkDeviceSize)dim * sizeof(float),        // Gamma
        (VkDeviceSize)rows * dim * sizeof(float), // G (upstream)
        (VkDeviceSize)rows * dim * sizeof(float), // DX (out)
        (VkDeviceSize)dim * sizeof(float),        // DGamma (atomic accumulator, uploaded zero)
        (VkDeviceSize)dim * sizeof(float),        // DBeta  (atomic accumulator, uploaded zero)
    };
    void* data[6] = { (void*)X, (void*)Gamma, (void*)G, DX, DGamma, DBeta };
    int up[6] = {1, 1, 1, 0, 1, 1}, down[6] = {0, 0, 0, 1, 1, 1};
    struct { int32_t rows; int32_t dim; float eps; } pc = { rows, dim, eps };
    // cooperative (§T347): ONE 256-invocation workgroup per row.
    return vk_dispatch(spv, spvLen, 6, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)rows, 1u, 1u);
}

int vk_unary_f32(const uint32_t* spv, int spvLen,
                 const float* X, float* O, int n, int op) {
    VkDeviceSize lens[2] = {
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
    };
    void* data[2] = { (void*)X, O };
    int up[2] = {1, 0}, down[2] = {0, 1};
    struct { int32_t n; int32_t op; } pc = { n, op };
    // one invocation per element; 256 elements per workgroup.
    return vk_dispatch(spv, spvLen, 2, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)n + 255u) / 256u, 1u, 1u);
}

// vk_batch_dispatch_bench (§T380, mirrors metal §T368): measure whether folding N dispatches into
// ONE command buffer (one submit + one vkQueueWaitIdle, barriers between) beats N per-op
// submit/waitIdle round-trips on this Vulkan/MoltenVK host — the de-risk BEFORE porting the whole
// recorder, since MoltenVK is bandwidth-bound and some Metal wins did not replicate here (§B39/B41).
// Runs the unary ReLU kernel `iters`× over one pooled buffer pair. oneBuffer=1 → batched; 0 → per-op.
int vk_batch_dispatch_bench(const uint32_t* spv, int spvLen, int iters, int oneBuffer, int n) {
    if (ensure_init() != 0) return -1;
    if (iters < 1 || n < 1) return -2;
    PipeCache* pc = NULL;
    struct { int32_t n; int32_t op; } push = { n, 4 /*ReLU*/ };
    if (pipeline_for(spv, spvLen, 2, sizeof(push), &pc) != 0) return -4;

    VkDeviceSize lens[2] = { (VkDeviceSize)n * sizeof(float), (VkDeviceSize)n * sizeof(float) };
    VkBuffer buf[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory mem[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    int poolIdx[2] = { -3, -3 };
    int rc = -2;
    for (int i = 0; i < 2; ++i) {
        poolIdx[i] = pool_acquire(lens[i], &buf[i], &mem[i]);
        if (poolIdx[i] == -2) { poolIdx[i] = -3; goto done; }
    }
    dset_write(pc->dset, 2, buf, lens);
    if (ensure_cmd() != 0) { rc = -5; goto done; }

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gCmd;
    uint32_t groups = ((uint32_t)n + 255u) / 256u;

    if (oneBuffer) {
        vkResetCommandBuffer(gCmd, 0);
        vkBeginCommandBuffer(gCmd, &cbbi);
        vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->pipe);
        vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->plLayout, 0, 1, &pc->dset, 0, NULL);
        vkCmdPushConstants(gCmd, pc->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        for (int it = 0; it < iters; ++it) {
            vkCmdDispatch(gCmd, groups, 1u, 1u);
            if (it + 1 < iters)
                vkCmdPipelineBarrier(gCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &mb, 0, NULL, 0, NULL);
        }
        vkEndCommandBuffer(gCmd);
        if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) { rc = -6; goto done; }
        if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) { rc = -6; goto done; }
    } else {
        for (int it = 0; it < iters; ++it) {
            vkResetCommandBuffer(gCmd, 0);
            vkBeginCommandBuffer(gCmd, &cbbi);
            vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->pipe);
            vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->plLayout, 0, 1, &pc->dset, 0, NULL);
            vkCmdPushConstants(gCmd, pc->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(gCmd, groups, 1u, 1u);
            vkEndCommandBuffer(gCmd);
            if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) { rc = -6; goto done; }
            if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) { rc = -6; goto done; }
        }
    }
    rc = 0;
done:
    for (int i = 0; i < 2; ++i) {
        if (poolIdx[i] == -3) continue;
        pool_release(poolIdx[i], buf[i], mem[i]);
    }
    return rc;
}

// vk_crossentropy_f32 (§T403): per-row loss L[b] = lse − z[target]; caller means over b. One
// 256-invocation workgroup per [b,c] row. Fixes the CPU-fallback loss forward (§T401 metal analog).
int vk_crossentropy_f32(const uint32_t* spv, int spvLen,
                        const float* Z, const float* T, float* L, int b, int c) {
    VkDeviceSize lens[3] = { (VkDeviceSize)b*c*sizeof(float), (VkDeviceSize)b*sizeof(float), (VkDeviceSize)b*sizeof(float) };
    void* data[3] = { (void*)Z, (void*)T, L };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t b; int32_t c; } pc = { b, c };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc), (uint32_t)b, 1u, 1u);
}

// vk_embed_f32 (§T403): row gather O[m,d], O[i,:]=Table[Idx[i],:]. One invocation per output elem.
int vk_embed_f32(const uint32_t* spv, int spvLen,
                 const float* Table, const float* Idx, float* O, int n, int m, int d) {
    VkDeviceSize lens[3] = { (VkDeviceSize)n*d*sizeof(float), (VkDeviceSize)m*sizeof(float), (VkDeviceSize)m*d*sizeof(float) };
    void* data[3] = { (void*)Table, (void*)Idx, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t m; int32_t d; } pc = { m, d };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc), ((uint32_t)(m*d) + 255u) / 256u, 1u, 1u);
}

int vk_binary_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* O, int n, int op) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
    };
    void* data[3] = { (void*)A, (void*)B, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t n; int32_t op; } pc = { n, op };
    // one invocation per element; 256 elements per workgroup.
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)n + 255u) / 256u, 1u, 1u);
}

// vk_addbias_f32 computes O[i,j] = X[rows,n][i,j] + B[n][j] (§T352). One invocation/elem.
int vk_addbias_f32(const uint32_t* spv, int spvLen,
                   const float* X, const float* B, float* O, int rows, int n) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)rows * n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)rows * n * sizeof(float),
    };
    void* data[3] = { (void*)X, (void*)B, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t rows; int32_t n; } pc = { rows, n };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(rows * n) + 255u) / 256u, 1u, 1u);
}

// vk_gelu_backward_f32 computes DX[i] = G[i]·gelu'(X[i]) over n elements (§T353).
int vk_gelu_backward_f32(const uint32_t* spv, int spvLen,
                         const float* X, const float* G, float* DX, int n) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
    };
    void* data[3] = { (void*)X, (void*)G, DX };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t n; } pc = { n };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)n + 255u) / 256u, 1u, 1u);
}

// vk_silu_backward_f32 computes DX[i] = G[i]·silu'(X[i]) over n elements (§T362).
int vk_silu_backward_f32(const uint32_t* spv, int spvLen,
                         const float* X, const float* G, float* DX, int n) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
    };
    void* data[3] = { (void*)X, (void*)G, DX };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    struct { int32_t n; } pc = { n };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)n + 255u) / 256u, 1u, 1u);
}

// vk_addbias_backward_f32 computes DB[j] = Σ_i G[rows,n][i,j] (§T354). One invocation/col.
int vk_addbias_backward_f32(const uint32_t* spv, int spvLen,
                            const float* G, float* DB, int rows, int n) {
    VkDeviceSize lens[2] = {
        (VkDeviceSize)rows * n * sizeof(float),
        (VkDeviceSize)n * sizeof(float),
    };
    void* data[2] = { (void*)G, DB };
    int up[2] = {1, 0}, down[2] = {0, 1};
    struct { int32_t rows; int32_t n; } pc = { rows, n };
    return vk_dispatch(spv, spvLen, 2, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)n + 255u) / 256u, 1u, 1u);
}

// push-constant block shared by the attention forward and backward kernels; must
// match shaders/mha.comp and shaders/mha_bwd.comp (seven int32 then one float).
typedef struct { int32_t sq, sk, dm, heads, kvHeads, dk, causal; float scale; } MhaPC;

// the forward kernel additionally takes a sliding-window bound (§T115) and a query-view
// element offset qBase (fused QKV, SPEC T613); must match shaders/mha.comp.
typedef struct { int32_t sq, sk, dm, heads, kvHeads, dk, causal, window, qBase; float scale; } MhaFwdPC;

// the backward kernel (shaders/mha_bwd.comp) keeps the pre-T613 layout: window but no qBase.
typedef struct { int32_t sq, sk, dm, heads, kvHeads, dk, causal, window; float scale; } MhaBwdPC;

int vk_mha_f32(const uint32_t* spv, int spvLen,
               const float* Q, const float* K, const float* V, float* O,
               int sq, int sk, int dm, int heads, int kvHeads, int dk,
               int causal, int window, float scale) {
    int dkv = kvHeads * dk;
    VkDeviceSize q = (VkDeviceSize)sq * dm * sizeof(float);
    VkDeviceSize kv = (VkDeviceSize)sk * dkv * sizeof(float);
    VkDeviceSize lens[4] = { q, kv, kv, q };
    void* data[4] = { (void*)Q, (void*)K, (void*)V, O };
    int up[4] = {1, 1, 1, 0}, down[4] = {0, 0, 0, 1};
    MhaFwdPC pc = { sq, sk, dm, heads, kvHeads, dk, causal, window, 0, scale };
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(heads * sq) + 63u) / 64u, 1u, 1u);
}

// vk_mha_decode_f32 (§T432): the cooperative decode/prefill attention (§T429/431 subgroup kernel)
// over host slices — the per-op path's route off the ~serial two-pass kernel at small sq.
int vk_mha_decode_f32(const uint32_t* spv, int spvLen,
                      const float* Q, const float* K, const float* V, float* O,
                      int sq, int sk, int dm, int heads, int kvHeads, int dk,
                      int causal, float scale) {
    int dkv = kvHeads * dk;
    VkDeviceSize q = (VkDeviceSize)sq * dm * sizeof(float);
    VkDeviceSize kv = (VkDeviceSize)sk * dkv * sizeof(float);
    VkDeviceSize lens[4] = { q, kv, kv, q };
    void* data[4] = { (void*)Q, (void*)K, (void*)V, O };
    int up[4] = {1, 1, 1, 0}, down[4] = {0, 0, 0, 1};
    struct { int32_t sq, sk, dm, heads, kvHeads, dk, causal, qBase; float scale; } pc = { sq, sk, dm, heads, kvHeads, dk, causal, 0, scale };
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       (uint32_t)heads, (uint32_t)sq, 1u);
}

int vk_mha_backward_f32(const uint32_t* spv, int spvLen,
                        const float* Q, const float* K, const float* V, const float* dO,
                        float* dQ, float* dK, float* dV,
                        int sq, int sk, int dm, int heads, int kvHeads, int dk,
                        int causal, int window, float scale) {
    if (ensure_init() != 0) return -1;
    if (!gAtomicFloat) return -7; // caller falls back to the reference (§I4)
    int dkv = kvHeads * dk;
    VkDeviceSize q = (VkDeviceSize)sq * dm * sizeof(float);
    VkDeviceSize kv = (VkDeviceSize)sk * dkv * sizeof(float);
    // bindings 0 Q, 1 K, 2 V, 3 dO, 4 dQ, 5 dK, 6 dV; dQ/dK/dV uploaded pre-zeroed
    VkDeviceSize lens[7] = { q, kv, kv, q, q, kv, kv };
    void* data[7] = { (void*)Q, (void*)K, (void*)V, (void*)dO, dQ, dK, dV };
    int up[7] = {1, 1, 1, 1, 1, 1, 1}, down[7] = {0, 0, 0, 0, 1, 1, 1};
    MhaBwdPC pc = { sq, sk, dm, heads, kvHeads, dk, causal, window, scale };
    return vk_dispatch(spv, spvLen, 7, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(heads * sq) + 63u) / 64u, 1u, 1u);
}

// push-constant block for the FlashAttention forward kernel; must match
// shaders/flashattn.comp (five int32 then one float).
typedef struct { int32_t seq, dm, heads, dk, causal, kvHeads; float scale; } FlashPC;

int vk_flashattn_f32(const uint32_t* spv, int spvLen,
                     const float* Q, const float* K, const float* V, float* O,
                     int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale) {
    VkDeviceSize qo = (VkDeviceSize)seq * dm * sizeof(float);            // Q,O [seq,dm]
    VkDeviceSize kv = (VkDeviceSize)seq * kvHeads * dk * sizeof(float);  // K,V [seq,kvHeads*dk]
    VkDeviceSize lens[4] = { qo, kv, kv, qo };
    void* data[4] = { (void*)Q, (void*)K, (void*)V, O };
    int up[4] = {1, 1, 1, 0}, down[4] = {0, 0, 0, 1};
    FlashPC pc = { seq, dm, heads, dk, causal, kvHeads, scale };
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(heads * seq) + 63u) / 64u, 1u, 1u);
}

// RetNet retention push-constant, shared by the value-expanded forward (§T179) and backward (§T192):
// Q,K,dQ,dK are [L,dk], V,O,dO,dV are [L,dv] (dv≠dk supported).
typedef struct { int32_t L, dk, dv; float gamma; } RetentionExpPC;

int vk_retention_f32(const uint32_t* spv, int spvLen,
                     const float* Q, const float* K, const float* V, float* O,
                     int L, int dk, int dv, float gamma) {
    VkDeviceSize szqk = (VkDeviceSize)L * dk * sizeof(float); // Q,K are [L,dk]
    VkDeviceSize szvo = (VkDeviceSize)L * dv * sizeof(float);  // V,O are [L,dv]
    VkDeviceSize lens[4] = { szqk, szqk, szvo, szvo };
    void* data[4] = { (void*)Q, (void*)K, (void*)V, O };
    int up[4] = {1, 1, 1, 0}, down[4] = {0, 0, 0, 1};
    RetentionExpPC pc = { L, dk, dv, gamma };
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)L + 63u) / 64u, 1u, 1u);
}

// RetNet retention backward (§T175/§T192): (Q,K [L,dk], V,dO [L,dv]) → (dQ,dK [L,dk], dV [L,dv]).
// Value expansion dv≠dk supported. One invocation per query row; dK/dV accumulated via float atomics
// (uploaded pre-zeroed).
int vk_retention_backward_f32(const uint32_t* spv, int spvLen,
                              const float* Q, const float* K, const float* V, const float* dO,
                              float* dQ, float* dK, float* dV, int L, int dk, int dv, float gamma) {
    if (ensure_init() != 0) return -1;
    if (!gAtomicFloat) return -7; // caller falls back to the reference (§I4)
    VkDeviceSize szqk = (VkDeviceSize)L * dk * sizeof(float); // Q,K,dQ,dK are [L,dk]
    VkDeviceSize szvo = (VkDeviceSize)L * dv * sizeof(float);  // V,dO,dV are [L,dv]
    // bindings 0 Q,1 K,2 V,3 dO,4 dQ,5 dK,6 dV; dQ/dK/dV uploaded pre-zeroed
    VkDeviceSize lens[7] = { szqk, szqk, szvo, szvo, szqk, szqk, szvo };
    void* data[7] = { (void*)Q, (void*)K, (void*)V, (void*)dO, dQ, dK, dV };
    int up[7] = {1, 1, 1, 1, 1, 1, 1}, down[7] = {0, 0, 0, 0, 1, 1, 1};
    RetentionExpPC pc = { L, dk, dv, gamma };
    return vk_dispatch(spv, spvLen, 7, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)L + 63u) / 64u, 1u, 1u);
}

// push-constant block for the quantized matmul kernels; a generic {M,K,N} shared by the Q8_0
// and Q4_K shaders (qmatmul_q8.comp / qmatmul_q4k.comp).
typedef struct { int32_t M, K, N; } QMatPC;

// vk_qmatmul_bytes is the shared byte-buffer quantized-matmul dispatch: X (f32) · dequant(W
// bytes)ᵀ → O (f32), with the in-kernel dequant living entirely in the SPIR-V module `spv`.
static int vk_qmatmul_bytes(const uint32_t* spv, int spvLen,
                            const float* X, const unsigned char* W, float* O,
                            int M, int K, int N, int wBytes) {
    VkDeviceSize lens[3] = {
        (VkDeviceSize)M * K * sizeof(float),
        (VkDeviceSize)wBytes,
        (VkDeviceSize)M * N * sizeof(float),
    };
    void* data[3] = { (void*)X, (void*)W, O };
    int up[3] = {1, 1, 0}, down[3] = {0, 0, 1};
    QMatPC pc = { M, K, N };
    return vk_dispatch(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(M * N) + 63u) / 64u, 1u, 1u);
}

int vk_qmatmul_q8_0(const uint32_t* spv, int spvLen,
                    const float* X, const unsigned char* W, float* O,
                    int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q4_0(const uint32_t* spv, int spvLen,
                    const float* X, const unsigned char* W, float* O,
                    int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q4k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q6k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q5k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q2k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

int vk_qmatmul_q3k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes) {
    return vk_qmatmul_bytes(spv, spvLen, X, W, O, M, K, N, wBytes);
}

// Device-resident quantized weights (§T156): a persistent VkBuffer holding a weight blob, reused
// across matmuls (uploaded once) instead of re-created per call. The handle bundles the buffer and
// its memory. wBytes must be a multiple of 4 (the shaders read the weight as a uint word array).
// (typedef hoisted above vk_coopmat_gemm_f16_res, which references it)

void* vk_qweight_upload(const unsigned char* W, int wBytes) {
    pthread_mutex_lock(&gLock);
    ResidentBuf* r = NULL;
    if (ensure_init() == 0 && (r = (ResidentBuf*)malloc(sizeof(ResidentBuf))) != NULL) {
        r->buf = VK_NULL_HANDLE;
        r->mem = VK_NULL_HANDLE;
        if (make_buffer((VkDeviceSize)wBytes, &r->buf, &r->mem) != 0) {
            if (r->buf) vkDestroyBuffer(gDevice, r->buf, NULL);
            if (r->mem) vkFreeMemory(gDevice, r->mem, NULL);
            free(r);
            r = NULL;
        } else {
            upload(r->mem, W, (size_t)wBytes);
        }
    }
    pthread_mutex_unlock(&gLock);
    return r;
}

void vk_qweight_free(void* handle) {
    if (!handle) return;
    ResidentBuf* r = (ResidentBuf*)handle;
    pthread_mutex_lock(&gLock);
    if (r->buf) vkDestroyBuffer(gDevice, r->buf, NULL);
    if (r->mem) vkFreeMemory(gDevice, r->mem, NULL);
    pthread_mutex_unlock(&gLock);
    free(r);
}

// ---- device-resident buffers for the Vulkan recorder (§T381, mirrors Metal §T367) ----------
// A persistent host-visible|coherent VkBuffer the batched-decode recorder chains dispatches over
// so intermediates stay resident instead of a host round-trip per op. nbytes bounds up/download.
typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize nbytes; } DevBuf;

void* vk_devbuf_upload(const void* data, int nbytes) {
    if (nbytes <= 0) return NULL;
    pthread_mutex_lock(&gLock);
    DevBuf* d = NULL;
    if (ensure_init() == 0 && (d = (DevBuf*)malloc(sizeof(DevBuf))) != NULL) {
        d->buf = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
        d->nbytes = (VkDeviceSize)nbytes;
        if (make_buffer((VkDeviceSize)nbytes, &d->buf, &d->mem) != 0) {
            if (d->buf) vkDestroyBuffer(gDevice, d->buf, NULL);
            if (d->mem) vkFreeMemory(gDevice, d->mem, NULL);
            free(d);
            d = NULL;
        } else if (data) {
            upload(d->mem, data, (size_t)nbytes);
        }
    }
    pthread_mutex_unlock(&gLock);
    return d;
}

int vk_devbuf_download(void* handle, void* dst, int nbytes) {
    if (!handle || nbytes <= 0) return -2;
    DevBuf* d = (DevBuf*)handle;
    if ((VkDeviceSize)nbytes > d->nbytes) return -3;
    download(d->mem, dst, (size_t)nbytes);
    return 0;
}

int vk_devbuf_upload_into(void* handle, const void* src, int nbytes) {
    if (!handle || nbytes <= 0) return -2;
    DevBuf* d = (DevBuf*)handle;
    if ((VkDeviceSize)nbytes > d->nbytes) return -3;
    upload(d->mem, src, (size_t)nbytes);
    return 0;
}

void vk_devbuf_free(void* handle) {
    if (!handle) return;
    DevBuf* d = (DevBuf*)handle;
    pthread_mutex_lock(&gLock);
    if (d->buf) vkDestroyBuffer(gDevice, d->buf, NULL);
    if (d->mem) vkFreeMemory(gDevice, d->mem, NULL);
    pthread_mutex_unlock(&gLock);
    free(d);
}

// ---- Vulkan recorder (§T382, mirrors Metal §T370) --------------------------------------------
// One open command buffer records dispatches over DevBufs with EXPLICIT barriers (Vulkan has no
// auto hazard tracking like Metal), submitted once with a single vkQueueWaitIdle at finish. A
// dedicated descriptor pool gives each recorded op its OWN set — a reused set would, at submit
// time, bind only its LAST write, corrupting all earlier dispatches. Single active recorder
// (decode is single-threaded here); each C call is gLock-guarded for queue/pool safety.
#define VK_REC_MAX_SETS 512
static VkCommandBuffer gRecCmd = VK_NULL_HANDLE;
static VkDescriptorPool gRecDescPool = VK_NULL_HANDLE;
static int gRecOpCount = 0;

static int ensure_rec_cmd(void) {
    if (gRecCmd) return 0;
    if (ensure_cmd() != 0) return -1; // creates gCmdPool
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = gCmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(gDevice, &cbai, &gRecCmd) != VK_SUCCESS) return -1;
    return 0;
}

static int ensure_rec_desc_pool(void) {
    if (gRecDescPool) return 0;
    VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * VK_REC_MAX_SETS };
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = VK_REC_MAX_SETS;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    if (vkCreateDescriptorPool(gDevice, &dpci, NULL, &gRecDescPool) != VK_SUCCESS) return -1;
    return 0;
}

void* vk_recorder_begin(void) {
    if (ensure_init() != 0) return NULL;
    pthread_mutex_lock(&gLock);
    void* tok = NULL;
    if (ensure_rec_cmd() == 0 && ensure_rec_desc_pool() == 0) {
        vkResetCommandBuffer(gRecCmd, 0);
        vkResetDescriptorPool(gDevice, gRecDescPool, 0); // free all sets from the prior recording
        VkCommandBufferBeginInfo cbbi = {0};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(gRecCmd, &cbbi) == VK_SUCCESS) {
            gRecOpCount = 0;
            tok = (void*)gRecCmd; // non-null token
        }
    }
    pthread_mutex_unlock(&gLock);
    return tok;
}

// rec_barrier inserts a broad compute+transfer memory barrier before an op (except the first),
// standing in for Metal's automatic hazard tracking. Covers RAW+WAR+WAW across BOTH compute
// dispatches and blit copies (src includes READ for WAR; TRANSFER stage/access for vkCmdCopyBuffer
// so a matmul→blit or blit→attention chain orders correctly). Caller holds gLock.
static void rec_barrier(void) {
    if (gRecOpCount <= 0) return;
    VkAccessFlags acc = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = acc;
    mb.dstAccessMask = acc;
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(gRecCmd, st, st, 0, 1, &mb, 0, NULL, 0, NULL);
}

// rec_dispatch allocates + writes a fresh descriptor set, records a barrier before the op
// (except the first), then binds pipeline+set+push and dispatches. Caller holds gLock.
static int rec_dispatch(PipeCache* pc, int nbuf, const VkBuffer* buf, const VkDeviceSize* lens,
                        const void* push, uint32_t pushLen, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (gRecOpCount >= VK_REC_MAX_SETS) return -8;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = gRecDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pc->dsl;
    if (vkAllocateDescriptorSets(gDevice, &dsai, &set) != VK_SUCCESS) return -7;
    dset_write(set, nbuf, buf, lens);
    rec_barrier();
    vkCmdBindPipeline(gRecCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->pipe);
    vkCmdBindDescriptorSets(gRecCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pc->plLayout, 0, 1, &set, 0, NULL);
    if (pushLen) vkCmdPushConstants(gRecCmd, pc->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushLen, push);
    vkCmdDispatch(gRecCmd, gx, gy, gz);
    gRecOpCount++;
    return 0;
}

// vk_recorder_unary records O = f(op, X) over n f32 elements of the device buffers xh, oh into the
// recorder's command buffer (op selector matches shaders/unary.comp). No submit.
int vk_recorder_unary(void* rec, const uint32_t* spv, int spvLen, void* xh, void* oh, int n, int op) {
    if (!rec || !xh || !oh || n < 1) return -2;
    DevBuf* x = (DevBuf*)xh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    struct { int32_t n; int32_t op; } push = { n, op };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 2, sizeof(push), &pc) == 0) {
        VkBuffer buf[2] = { x->buf, o->buf };
        VkDeviceSize lens[2] = { (VkDeviceSize)n * 4, (VkDeviceSize)n * 4 };
        rc = rec_dispatch(pc, 2, buf, lens, &push, sizeof(push), ((uint32_t)n + 255u) / 256u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_binary records O = A (op) B over n f32 elements of device buffers (op selectors
// match shaders/binary.comp: 0=add,1=sub,2=mul,3=div,4=max,5=min). No submit.
int vk_recorder_binary(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* oh, int n, int op) {
    if (!rec || !ah || !bh || !oh || n < 1) return -2;
    DevBuf* a = (DevBuf*)ah; DevBuf* b = (DevBuf*)bh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    struct { int32_t n; int32_t op; } push = { n, op };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { a->buf, b->buf, o->buf };
        VkDeviceSize lens[3] = { (VkDeviceSize)n*4, (VkDeviceSize)n*4, (VkDeviceSize)n*4 };
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push), ((uint32_t)n + 255u) / 256u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_matmul records C = A·B (M×K · K×N → M×N) over device buffers via the tiled GEMM
// shader (16×16 output tiles). Forward-only (no transpose — decode needs none). No submit.
int vk_recorder_matmul(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* ch, int M, int K, int N,
                       int accumulate) {
    if (!rec || !ah || !bh || !ch || M < 1 || K < 1 || N < 1) return -2;
    DevBuf* a = (DevBuf*)ah; DevBuf* b = (DevBuf*)bh; DevBuf* c = (DevBuf*)ch;
    PipeCache* pc = NULL;
    // accumulate!=0: C += A·B — the residual-add epilogue (SPEC T613).
    int32_t push[6] = { M, K, N, 0, 0, accumulate }; // {M,K,N,transA,transB,acc}
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { a->buf, b->buf, c->buf };
        VkDeviceSize lens[3] = { (VkDeviceSize)M*K*4, (VkDeviceSize)K*N*4, (VkDeviceSize)M*N*4 };
        rc = rec_dispatch(pc, 3, buf, lens, push, sizeof(push),
                          ((uint32_t)N + 15u) / 16u, ((uint32_t)M + 15u) / 16u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_rmsnorm records O = rmsnorm(X)·gamma over device buffers (cooperative: one
// 256-invocation workgroup per row). No submit.
int vk_recorder_rmsnorm(void* rec, const uint32_t* spv, int spvLen, void* xh, void* gh, void* oh, int rows, int dim, float eps) {
    if (!rec || !xh || !gh || !oh || rows < 1 || dim < 1) return -2;
    DevBuf* x = (DevBuf*)xh; DevBuf* g = (DevBuf*)gh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    struct { int32_t rows; int32_t dim; float eps; } push = { rows, dim, eps };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { x->buf, g->buf, o->buf };
        VkDeviceSize lens[3] = { (VkDeviceSize)rows*dim*4, (VkDeviceSize)dim*4, (VkDeviceSize)rows*dim*4 };
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push), (uint32_t)rows, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_addbias records O[i,j] = X[i,j] + B[j] over device buffers (§T423, GPT StepN's
// broadcast bias-add). No submit.
int vk_recorder_addbias(void* rec, const uint32_t* spv, int spvLen, void* xh, void* bh, void* oh, int rows, int n) {
    if (!rec || !xh || !bh || !oh || rows < 1 || n < 1) return -2;
    DevBuf* x = (DevBuf*)xh; DevBuf* b = (DevBuf*)bh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    struct { int32_t rows; int32_t n; } push = { rows, n };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { x->buf, b->buf, o->buf };
        VkDeviceSize lens[3] = { (VkDeviceSize)rows*n*4, (VkDeviceSize)n*4, (VkDeviceSize)rows*n*4 };
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push), ((uint32_t)(rows*n) + 255u) / 256u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_layernorm records O = layernorm(X)·gamma + beta over device buffers (cooperative:
// one 256-invocation workgroup per row). GPT-2-style norm (rmsnorm is the Llama variant). No submit.
int vk_recorder_layernorm(void* rec, const uint32_t* spv, int spvLen, void* xh, void* gh, void* bh, void* oh, int rows, int dim, float eps) {
    if (!rec || !xh || !gh || !bh || !oh || rows < 1 || dim < 1) return -2;
    DevBuf* x = (DevBuf*)xh; DevBuf* g = (DevBuf*)gh; DevBuf* b = (DevBuf*)bh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    struct { int32_t rows; int32_t dim; float eps; } push = { rows, dim, eps };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 4, sizeof(push), &pc) == 0) {
        VkBuffer buf[4] = { x->buf, g->buf, b->buf, o->buf };
        VkDeviceSize lens[4] = { (VkDeviceSize)rows*dim*4, (VkDeviceSize)dim*4, (VkDeviceSize)dim*4, (VkDeviceSize)rows*dim*4 };
        rc = rec_dispatch(pc, 4, buf, lens, &push, sizeof(push), (uint32_t)rows, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_rope records O = RoPE(Q) over device buffers (one invocation per rotated pair,
// 64/workgroup). posOffset = the query's absolute start position (KV-cache length). No submit.
int vk_recorder_rope(void* rec, const uint32_t* spv, int spvLen, void* qh, void* invh, void* oh,
                     int seq, int width, int heads, int hd, int half, int posOffset, float posDiv,
                     int elemOff) {
    if (!rec || !qh || !invh || !oh || seq < 1 || width < 1 || elemOff < 0) return -2;
    DevBuf* q = (DevBuf*)qh; DevBuf* inv = (DevBuf*)invh; DevBuf* o = (DevBuf*)oh;
    PipeCache* pc = NULL;
    // elemOff is a float-element offset applied to Q AND O inside the shader (the fused-QKV
    // sub-row view, SPEC T613) — an index, not a descriptor offset, so no
    // minStorageBufferOffsetAlignment constraint. The bound ranges must cover it.
    struct { int32_t seq, width, heads, hd, half, posOffset, off; float posDiv; } push = {
        seq, width, heads, hd, half, posOffset, elemOff, posDiv };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { q->buf, inv->buf, o->buf };
        VkDeviceSize span = ((VkDeviceSize)elemOff + (VkDeviceSize)seq*width) * 4;
        VkDeviceSize lens[3] = { span, (VkDeviceSize)half*4, span };
        uint32_t total = (uint32_t)seq * (uint32_t)heads * (uint32_t)half;
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push), (total + 63u) / 64u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_rope2 records the fused two-band in-place rotation (SPEC T613): ONE dispatch
// rotates the q band (headsQ heads at element offset offQ) AND the k band (headsK heads at
// offK) of a fused QKV buffer, rows `stride` floats wide. Replaces two rope dispatches.
int vk_recorder_rope2(void* rec, const uint32_t* spv, int spvLen, void* qh, void* invh,
                      int seq, int stride, int headsQ, int offQ, int headsK, int offK,
                      int hd, int half, int posOffset, float posDiv) {
    if (!rec || !qh || !invh || seq < 1 || stride < 1 || headsQ < 1 || headsK < 1 ||
        offQ < 0 || offK < 0) return -2;
    DevBuf* q = (DevBuf*)qh; DevBuf* inv = (DevBuf*)invh;
    PipeCache* pc = NULL;
    struct { int32_t seq, stride, headsQ, offQ, headsK, offK, hd, half, posOffset; float posDiv; } push = {
        seq, stride, headsQ, offQ, headsK, offK, hd, half, posOffset, posDiv };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 2, sizeof(push), &pc) == 0) {
        int maxOff = offQ > offK ? offQ : offK;
        VkDeviceSize span = ((VkDeviceSize)maxOff + (VkDeviceSize)seq*stride) * 4;
        if (span > q->nbytes) span = q->nbytes;
        VkBuffer buf[2] = { q->buf, inv->buf };
        VkDeviceSize lens[2] = { span, (VkDeviceSize)half*4 };
        uint32_t total = (uint32_t)seq * (uint32_t)(headsQ + headsK) * (uint32_t)half;
        rc = rec_dispatch(pc, 2, buf, lens, &push, sizeof(push), (total + 63u) / 64u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_mha_decode (§T429, the vulkan twin of metal §T428): cooperative single-query decode
// attention — one 32-lane subgroup per head (vs the two-pass kernel's one invocation per head,
// which made decode ~serial in the context length). Push {sk,dm,heads,kvHeads,dk,scale}. No submit.
int vk_recorder_mha_decode(void* rec, const uint32_t* spv, int spvLen, void* qh, void* kh, void* vh, void* oh,
                           int sq, int sk, int dm, int heads, int kvHeads, int dk, int causal, float scale,
                           int qElemOff) {
    if (!rec || !qh || !kh || !vh || !oh || sq < 1 || sk < 1 || qElemOff < 0) return -2;
    DevBuf* q = (DevBuf*)qh; DevBuf* k = (DevBuf*)kh; DevBuf* v = (DevBuf*)vh; DevBuf* o = (DevBuf*)oh;
    int dkv = kvHeads * dk;
    PipeCache* pc = NULL;
    // qElemOff: float-element offset of the query view inside a fused QKV buffer (SPEC T613).
    struct { int32_t sq, sk, dm, heads, kvHeads, dk, causal, qBase; float scale; } push = { sq, sk, dm, heads, kvHeads, dk, causal, qElemOff, scale };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 4, sizeof(push), &pc) == 0) {
        VkBuffer buf[4] = { q->buf, k->buf, v->buf, o->buf };
        VkDeviceSize lens[4] = { ((VkDeviceSize)qElemOff + (VkDeviceSize)sq*dm)*4, (VkDeviceSize)sk*dkv*4, (VkDeviceSize)sk*dkv*4, (VkDeviceSize)sq*dm*4 };
        rc = rec_dispatch(pc, 4, buf, lens, &push, sizeof(push), (uint32_t)heads, (uint32_t)sq, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_qmatmul (§T414, mirrors metal §T413): record-mode QUANTIZED matmul — O = X·dequant(W)ᵀ
// encoded into the recorder's command buffer over DevBuf X/O and a RESIDENT quantized weight
// (vk_qweight_upload). The shader is the per-quant-type resident matmul spv. No submit.
int vk_recorder_qmatmul(void* rec, const uint32_t* spv, int spvLen, void* xh, void* wHandle, void* oh,
                        int M, int K, int N, int wBytes) {
    if (!rec || !xh || !wHandle || !oh || M < 1 || K < 1 || N < 1) return -2;
    DevBuf* x = (DevBuf*)xh; DevBuf* o = (DevBuf*)oh;
    ResidentBuf* w = (ResidentBuf*)wHandle;
    PipeCache* pc = NULL;
    QMatPC push = { M, K, N };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { x->buf, w->buf, o->buf };
        VkDeviceSize lens[3] = { (VkDeviceSize)M*K*4, (VkDeviceSize)wBytes, (VkDeviceSize)M*N*4 };
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push), ((uint32_t)(M*N) + 63u) / 64u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_mha records O = attention(Q,K,V) over device buffers with SEPARATE query/key
// lengths (sq queries vs sk cached keys — the decode-against-KV-cache shape sq=1, sk=cacheLen).
// K,V may be over-allocated caches; only the first sk rows are bound/read. No submit.
// vk_recorder_matmul_strided records C = alpha·A·(B/Bᵀ) with per-operand strides+offsets (the §T397
// attention matmuls over windows of packed Q/K/V/O buffers). No submit.
typedef struct {
    int32_t M, K, N, transA, transB, lda, ldb, ldc, offA, offB, offC;
    float alpha;
} StridedPC;
int vk_recorder_matmul_strided(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* ch,
                               int M, int K, int N, int transA, int transB,
                               int lda, int ldb, int ldc, int offA, int offB, int offC, float alpha) {
    if (!rec || !ah || !bh || !ch || M < 1 || K < 1 || N < 1) return -2;
    DevBuf* a = (DevBuf*)ah; DevBuf* b = (DevBuf*)bh; DevBuf* c = (DevBuf*)ch;
    PipeCache* pc = NULL;
    StridedPC push = { M, K, N, transA, transB, lda, ldb, ldc, offA, offB, offC, alpha };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 3, sizeof(push), &pc) == 0) {
        VkBuffer buf[3] = { a->buf, b->buf, c->buf };
        VkDeviceSize lens[3] = { a->nbytes, b->nbytes, c->nbytes };
        rc = rec_dispatch(pc, 3, buf, lens, &push, sizeof(push),
                          ((uint32_t)N + 15u) / 16u, ((uint32_t)M + 15u) / 16u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_softmax_causal records an in-place row softmax of S[rows,cols] with an optional causal
// mask (256-invocation workgroup per row). No submit.
int vk_recorder_softmax_causal(void* rec, const uint32_t* spv, int spvLen, void* sh, int rows, int cols, int causal) {
    if (!rec || !sh || rows < 1 || cols < 1) return -2;
    DevBuf* s = (DevBuf*)sh;
    PipeCache* pc = NULL;
    struct { int32_t rows, cols, causal; } push = { rows, cols, causal };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 1, sizeof(push), &pc) == 0) {
        VkBuffer buf[1] = { s->buf };
        VkDeviceSize lens[1] = { s->nbytes };
        rc = rec_dispatch(pc, 1, buf, lens, &push, sizeof(push), (uint32_t)rows, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

int vk_recorder_mha(void* rec, const uint32_t* spv, int spvLen, void* qh, void* kh, void* vh, void* oh,
                    int sq, int sk, int dm, int heads, int kvHeads, int dk, int causal, int window, float scale,
                    int qElemOff) {
    if (!rec || !qh || !kh || !vh || !oh || sq < 1 || sk < 1 || qElemOff < 0) return -2;
    DevBuf* q = (DevBuf*)qh; DevBuf* k = (DevBuf*)kh; DevBuf* v = (DevBuf*)vh; DevBuf* o = (DevBuf*)oh;
    int dkv = kvHeads * dk;
    PipeCache* pc = NULL;
    // qElemOff: float-element offset of the query view inside a fused QKV buffer (SPEC T613).
    MhaFwdPC push = { sq, sk, dm, heads, kvHeads, dk, causal, window, qElemOff, scale };
    pthread_mutex_lock(&gLock);
    int rc = -4;
    if (pipeline_for(spv, spvLen, 4, sizeof(push), &pc) == 0) {
        VkBuffer buf[4] = { q->buf, k->buf, v->buf, o->buf };
        VkDeviceSize qb = (VkDeviceSize)sq * dm * 4, kv = (VkDeviceSize)sk * dkv * 4;
        VkDeviceSize lens[4] = { (VkDeviceSize)qElemOff*4 + qb, kv, kv, qb };
        rc = rec_dispatch(pc, 4, buf, lens, &push, sizeof(push), ((uint32_t)(heads * sq) + 63u) / 64u, 1u, 1u);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_recorder_blit records a buffer→buffer copy of nbytes from src[srcOff] to dst[dstOff] into the
// recorder's command buffer (vkCmdCopyBuffer). The KV-cache APPEND primitive: copy the new k/v row
// into cache[cacheLen]. A rec_barrier before it orders a preceding matmul write before the copy.
int vk_recorder_blit(void* rec, void* srcH, int srcOff, void* dstH, int dstOff, int nbytes) {
    if (!rec || !srcH || !dstH || nbytes <= 0) return -2;
    DevBuf* s = (DevBuf*)srcH; DevBuf* d = (DevBuf*)dstH;
    if ((VkDeviceSize)(srcOff + nbytes) > s->nbytes || (VkDeviceSize)(dstOff + nbytes) > d->nbytes) return -2;
    pthread_mutex_lock(&gLock);
    rec_barrier();
    VkBufferCopy region = { (VkDeviceSize)srcOff, (VkDeviceSize)dstOff, (VkDeviceSize)nbytes };
    vkCmdCopyBuffer(gRecCmd, s->buf, d->buf, 1, &region);
    gRecOpCount++;
    pthread_mutex_unlock(&gLock);
    return 0;
}

// vk_recorder_copy2d records a strided rows×rowFloats sub-matrix copy src→dst: row r copies
// rowFloats f32 from src[srcOff + r·srcStride] to dst[dstOff + r·dstStride] (offsets/strides
// in ELEMENTS). ONE vkCmdCopyBuffer with `rows` regions — a single command, not rows commands.
// Extracts/deposits the q/k/v sub-columns of a fused QKV buffer at prefill (SPEC T613).
int vk_recorder_copy2d(void* rec, void* srcH, int srcOff, int srcStride,
                       void* dstH, int dstOff, int dstStride, int rows, int rowFloats) {
    if (!rec || !srcH || !dstH || rows < 1 || rowFloats < 1 ||
        srcOff < 0 || dstOff < 0 || srcStride < rowFloats || dstStride < rowFloats) return -2;
    DevBuf* s = (DevBuf*)srcH; DevBuf* d = (DevBuf*)dstH;
    VkDeviceSize sEnd = ((VkDeviceSize)srcOff + (VkDeviceSize)(rows-1)*srcStride + rowFloats) * 4;
    VkDeviceSize dEnd = ((VkDeviceSize)dstOff + (VkDeviceSize)(rows-1)*dstStride + rowFloats) * 4;
    if (sEnd > s->nbytes || dEnd > d->nbytes) return -2;
    VkBufferCopy* regions = (VkBufferCopy*)malloc(sizeof(VkBufferCopy) * (size_t)rows);
    if (!regions) return -3;
    for (int r = 0; r < rows; r++) {
        regions[r].srcOffset = ((VkDeviceSize)srcOff + (VkDeviceSize)r*srcStride) * 4;
        regions[r].dstOffset = ((VkDeviceSize)dstOff + (VkDeviceSize)r*dstStride) * 4;
        regions[r].size = (VkDeviceSize)rowFloats * 4;
    }
    pthread_mutex_lock(&gLock);
    rec_barrier();
    vkCmdCopyBuffer(gRecCmd, s->buf, d->buf, (uint32_t)rows, regions);
    gRecOpCount++;
    pthread_mutex_unlock(&gLock);
    free(regions);
    return 0;
}

int vk_recorder_finish(void* rec) {
    if (!rec) return -2;
    pthread_mutex_lock(&gLock);
    int rc = -6;
    if (vkEndCommandBuffer(gRecCmd) == VK_SUCCESS) {
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &gRecCmd;
        if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS &&
            vkQueueWaitIdle(gQueue) == VK_SUCCESS) {
            rc = 0;
        }
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

void vk_recorder_free(void* rec) { (void)rec; /* global recorder state; reset on next begin */ }

// vk_qmatmul_resident is vk_qmatmul_bytes but the weight is the already-resident buffer `wHandle`
// (from vk_qweight_upload); only X is uploaded. `spv` selects the per-quant-type shader.
int vk_qmatmul_resident(const uint32_t* spv, int spvLen,
                        const float* X, void* wHandle, float* O,
                        int M, int K, int N, int wBytes) {
    if (!wHandle) return -2;
    ResidentBuf* r = (ResidentBuf*)wHandle;
    VkDeviceSize lens[3] = {
        (VkDeviceSize)M * K * sizeof(float),
        (VkDeviceSize)wBytes,
        (VkDeviceSize)M * N * sizeof(float),
    };
    void* data[3] = { (void*)X, NULL, O };
    int up[3] = {1, 0, 0}, down[3] = {0, 0, 1};
    VkBuffer preBuf[3] = { VK_NULL_HANDLE, r->buf, VK_NULL_HANDLE };
    QMatPC pc = { M, K, N };
    return vk_dispatch_pre(spv, spvLen, 3, lens, data, up, down, &pc, sizeof(pc),
                           ((uint32_t)(M * N) + 63u) / 64u, 1u, 1u, preBuf);
}

// push-constant block shared by the conv2d im2col/colout kernels (must match
// shaders/im2col.comp + shaders/colout.comp) and the conv2d backward kernel
// (shaders/conv2d_bwd.comp).
typedef struct { int32_t N, C, H, Wd, F, KH, KW, stride, pad, ho, wo; } ConvPC;

// vk_conv2d_f32 lowers the forward convolution to im2col + the tiled GEMM shader +
// a scatter/bias pass (§T342, the Vulkan twin of the Metal §T341 path; the naive
// one-invocation-per-output kernel peaked at ~250 GFLOP/s vs the tiled GEMM's ~590):
//   col[c·KH·KW+ky·KW+kx, (n·ho+oy)·wo+ox] = X[...] (0 in padding)   [im2col.comp]
//   G = W[F, C·KH·KW] · col                                          [matmul.comp]
//   Out[n,f,oy,ox] = G[f, (n·ho+oy)·wo+ox] + Bias[f]                 [colout.comp]
// The three stages run on device-resident pooled buffers under one gLock hold; each
// vk_dispatch_locked waits the queue idle, which orders the stages. Every element
// of col/G/Out is written, so pooled reuse stays stale-safe.
int vk_conv2d_f32(const uint32_t* spvIm2col, int im2colLen,
                  const uint32_t* spvMatmul, int matmulLen,
                  const uint32_t* spvColout, int coloutLen,
                  const float* X, const float* W, const float* B, float* Out,
                  int N, int C, int H, int Wd, int F, int KH, int KW,
                  int stride, int pad, int ho, int wo) {
    pthread_mutex_lock(&gLock);
    int rc = -1;
    if (ensure_init() != 0) goto unlock;

    {
        int rows = C * KH * KW;   // GEMM interior dim
        int cols = N * ho * wo;   // GEMM result columns
        VkDeviceSize xLen = (VkDeviceSize)N * C * H * Wd * sizeof(float);
        VkDeviceSize wLen = (VkDeviceSize)F * rows * sizeof(float);
        VkDeviceSize bLen = (VkDeviceSize)F * sizeof(float);
        VkDeviceSize colLen = (VkDeviceSize)rows * cols * sizeof(float);
        VkDeviceSize gLen = (VkDeviceSize)F * cols * sizeof(float);
        VkDeviceSize oLen = (VkDeviceSize)N * F * ho * wo * sizeof(float);

        VkBuffer buf[6];
        VkDeviceMemory mem[6];
        int idx[6];
        VkDeviceSize need[6] = { xLen, wLen, bLen, colLen, gLen, oLen };
        for (int i = 0; i < 6; ++i) { buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE; idx[i] = -3; }
        rc = -2;
        for (int i = 0; i < 6; ++i) {
            idx[i] = pool_acquire(need[i], &buf[i], &mem[i]);
            if (idx[i] == -2) { idx[i] = -3; goto done; }
        }
        upload(mem[0], X, (size_t)xLen);
        upload(mem[1], W, (size_t)wLen);
        upload(mem[2], B, (size_t)bLen);

        ConvPC pc = { N, C, H, Wd, F, KH, KW, stride, pad, ho, wo };
        int32_t mpc[6] = { F, rows, cols, 0, 0, 0 }; // matmul push {M,K,N,transA,transB,acc} (SPEC T356, T613)

        // All three stages record into ONE command buffer with compute→compute
        // memory barriers ordering the writes (§T343) — one submit + one wait
        // instead of a submit/waitIdle round trip per stage.
        PipeCache *p1 = NULL, *p2 = NULL, *p3 = NULL;
        rc = -4;
        if (pipeline_for(spvIm2col, im2colLen, 2, sizeof(pc), &p1) != 0) goto done;
        if (pipeline_for(spvMatmul, matmulLen, 3, sizeof(mpc), &p2) != 0) goto done;
        if (pipeline_for(spvColout, coloutLen, 3, sizeof(pc), &p3) != 0) goto done;

        {   // rewrite the three cached descriptor sets (queue idle → legal)
            VkDeviceSize l1[2] = { xLen, colLen };
            VkBuffer b1[2] = { buf[0], buf[3] };
            dset_write(p1->dset, 2, b1, l1);
            VkDeviceSize l2[3] = { wLen, colLen, gLen };
            VkBuffer b2[3] = { buf[1], buf[3], buf[4] };
            dset_write(p2->dset, 3, b2, l2);
            VkDeviceSize l3[3] = { gLen, bLen, oLen };
            VkBuffer b3[3] = { buf[4], buf[2], buf[5] };
            dset_write(p3->dset, 3, b3, l3);
        }

        if (ensure_cmd() != 0) { rc = -5; goto done; }
        vkResetCommandBuffer(gCmd, 0);
        VkCommandBufferBeginInfo cbbi = {0};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(gCmd, &cbbi);

        VkMemoryBarrier mb = {0};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p1->pipe);
        vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p1->plLayout, 0, 1, &p1->dset, 0, NULL);
        vkCmdPushConstants(gCmd, p1->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(gCmd, ((uint32_t)(rows * cols) + 255u) / 256u, 1u, 1u);
        vkCmdPipelineBarrier(gCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);

        vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p2->pipe);
        vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p2->plLayout, 0, 1, &p2->dset, 0, NULL);
        vkCmdPushConstants(gCmd, p2->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), mpc);
        vkCmdDispatch(gCmd, ((uint32_t)cols + 15u) / 16u, ((uint32_t)F + 15u) / 16u, 1u);
        vkCmdPipelineBarrier(gCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);

        vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p3->pipe);
        vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, p3->plLayout, 0, 1, &p3->dset, 0, NULL);
        vkCmdPushConstants(gCmd, p3->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(gCmd, ((uint32_t)(N * F * ho * wo) + 255u) / 256u, 1u, 1u);
        vkEndCommandBuffer(gCmd);

        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &gCmd;
        rc = -6;
        if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) goto done;
        if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) goto done;

        download(mem[5], Out, (size_t)oLen);
        rc = 0;
done:
        for (int i = 0; i < 6; ++i) {
            if (idx[i] == -3) continue;
            pool_release(idx[i], buf[i], mem[i]);
        }
    }
unlock:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_conv2d_igemm_f32 is the FUSED implicit-GEMM forward conv (§T620): one dispatch
// of conv_igemm.comp, no im2col/G intermediate buffers. The kernel's B-panel load
// gathers X directly via the im2col index and the epilogue scatters+biases straight
// to NCHW, so only X, W, Bias (uploaded) and Out (downloaded) touch global memory —
// killing the O(N·C·K²·HW) col write+read that dominated the im2col path. The GEMM
// dims (M=F, K=C·KH·KW, Ncol=N·ho·wo) are derived in-shader from ConvPC. The kernel
// is N-blocked (NB=4 output cols/thread, vec4 gather §T620 coalescing rung), so a
// 16×16 workgroup covers a 64-wide column tile. Dispatch: ceil(Ncol/64) × ceil(F/16)
// workgroups. Returns 0 on success, nonzero on error.
int vk_conv2d_igemm_f32(const uint32_t* spv, int spvLen,
                        const float* X, const float* W, const float* B, float* Out,
                        int N, int C, int H, int Wd, int F, int KH, int KW,
                        int stride, int pad, int ho, int wo) {
    int K    = C * KH * KW;   // GEMM interior
    int Ncol = N * ho * wo;   // GEMM columns
    VkDeviceSize lens[4] = {
        (VkDeviceSize)N * C * H * Wd * sizeof(float),   // X[N,C,H,Wd]
        (VkDeviceSize)F * K * sizeof(float),            // W[F, C·KH·KW]
        (VkDeviceSize)F * sizeof(float),                // Bias[F]
        (VkDeviceSize)N * F * ho * wo * sizeof(float),  // Out[N,F,ho,wo]
    };
    void* data[4] = { (void*)X, (void*)W, (void*)B, Out };
    int up[4]   = {1, 1, 1, 0};
    int down[4] = {0, 0, 0, 1};
    ConvPC pc = { N, C, H, Wd, F, KH, KW, stride, pad, ho, wo };
    return vk_dispatch(spv, spvLen, 4, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)Ncol + 63u) / 64u, ((uint32_t)F + 15u) / 16u, 1u);
}

int vk_conv2d_backward_f32(const uint32_t* spv, int spvLen,
                           const float* X, const float* W, const float* dO,
                           float* dX, float* dW, float* dBias,
                           int N, int C, int H, int Wd, int F, int KH, int KW,
                           int stride, int pad, int ho, int wo) {
    if (ensure_init() != 0) return -1;
    if (!gAtomicFloat) return -7; // caller falls back to the reference (§I4)
    VkDeviceSize xLen = (VkDeviceSize)N * C * H * Wd * sizeof(float);
    VkDeviceSize wLen = (VkDeviceSize)F * C * KH * KW * sizeof(float);
    VkDeviceSize oLen = (VkDeviceSize)N * F * ho * wo * sizeof(float);
    VkDeviceSize bLen = (VkDeviceSize)F * sizeof(float);
    // bindings 0 X, 1 W, 2 dO, 3 dX, 4 dW, 5 dBias; dX/dW/dBias uploaded pre-zeroed
    VkDeviceSize lens[6] = { xLen, wLen, oLen, xLen, wLen, bLen };
    void* data[6] = { (void*)X, (void*)W, (void*)dO, dX, dW, dBias };
    int up[6] = {1, 1, 1, 1, 1, 1}, down[6] = {0, 0, 0, 1, 1, 1};
    ConvPC pc = { N, C, H, Wd, F, KH, KW, stride, pad, ho, wo };
    return vk_dispatch(spv, spvLen, 6, lens, data, up, down, &pc, sizeof(pc),
                       ((uint32_t)(N * F * ho * wo) + 63u) / 64u, 1u, 1u);
}

// vk_mha_backward_chain (§T528): the matmul-decomposed SDPA backward — the vulkan
// analog of metal's §T399 MPS path. Seven GPU stages over PACKED per-head buffers:
//   1. S_h = scale·Q_h·K_hᵀ          (matmul_strided ×heads, one dset)
//   2. P = softmax_packed(S)          (causal via row % seq)
//   3. dP_h = dO_h·V_hᵀ               (matmul_strided ×heads)
//   4. dS = scale·P⊙(dP − rowsum)     (sm_jacobian, in place in dP)
//   5. dQ_h = dS_h·K_h
//   6. dVt_h = P_hᵀ·dO_h              (per QUERY head; caller sums GQA groups)
//   7. dKt_h = dS_hᵀ·Q_h              (per QUERY head; caller sums GQA groups)
// Each stage is one submit (the cached per-pipeline descriptor set is rewritten
// between submits — queue idle makes that legal, §T343 note); per-head windows are
// addressed with the strided shader's offsets, so a stage needs only ONE dset.
int vk_mha_backward_chain(const uint32_t* spvMatmul, int matmulLen,
                          const uint32_t* spvSoftmax, int softmaxLen,
                          const uint32_t* spvJac, int jacLen,
                          const float* Q, const float* K, const float* V, const float* dO,
                          float* dQ, float* dVt, float* dKt,
                          int seq, int dm, int heads, int kvHeads, int dk,
                          int causal, float scale) {
    if (ensure_init() != 0) return -1;
    pthread_mutex_lock(&gLock);
    int rc = -2;
    int dkv = kvHeads * dk;
    VkDeviceSize qLen = (VkDeviceSize)seq * dm * sizeof(float);
    VkDeviceSize kLen = (VkDeviceSize)seq * dkv * sizeof(float);
    VkDeviceSize sLen = (VkDeviceSize)heads * seq * seq * sizeof(float);
    VkDeviceSize tLen = (VkDeviceSize)heads * seq * dk * sizeof(float);
    // 0 Q, 1 K, 2 V, 3 dO, 4 S/P, 5 dP/dS, 6 dQ, 7 dVt, 8 dKt
    VkDeviceSize lens[9] = { qLen, kLen, kLen, qLen, sLen, sLen, qLen, tLen, tLen };
    VkBuffer buf[9]; VkDeviceMemory mem[9]; int idx[9];
    for (int i = 0; i < 9; ++i) { idx[i] = -3; buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE; }
    for (int i = 0; i < 9; ++i) {
        idx[i] = pool_acquire(lens[i], &buf[i], &mem[i]);
        if (idx[i] < -1) goto done;
    }
    upload(mem[0], Q, (size_t)qLen);
    upload(mem[1], K, (size_t)kLen);
    upload(mem[2], V, (size_t)kLen);
    upload(mem[3], dO, (size_t)qLen);

    {
        PipeCache *pm = NULL, *ps = NULL, *pj = NULL;
        rc = -4;
        if (pipeline_for(spvMatmul, matmulLen, 3, sizeof(StridedPC), &pm) != 0) goto done;
        if (pipeline_for(spvSoftmax, softmaxLen, 1, 4 * sizeof(int32_t), &ps) != 0) goto done;
        if (pipeline_for(spvJac, jacLen, 2, 3 * sizeof(int32_t), &pj) != 0) goto done;

        uint32_t gx = ((uint32_t)seq + 15u) / 16u, gy = ((uint32_t)seq + 15u) / 16u;
        uint32_t gxd = ((uint32_t)dk + 15u) / 16u;
        VkMemoryBarrier mb = {0};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        // one submit running `n` strided matmuls (push varies per head, dset fixed)
        #define SUBMIT_MATMULS(pcArr, n, GX, GY)                                              \
            do {                                                                              \
                if (ensure_cmd() != 0) { rc = -5; goto done; }                                \
                vkResetCommandBuffer(gCmd, 0);                                               \
                VkCommandBufferBeginInfo cbbi = {0};                                          \
                cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;                     \
                cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;                     \
                vkBeginCommandBuffer(gCmd, &cbbi);                                            \
                vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->pipe);            \
                vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->plLayout,   \
                                        0, 1, &pm->dset, 0, NULL);                            \
                for (int h = 0; h < (n); ++h) {                                               \
                    vkCmdPushConstants(gCmd, pm->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,    \
                                       sizeof(StridedPC), &(pcArr)[h]);                       \
                    vkCmdDispatch(gCmd, (GX), (GY), 1u);                                      \
                }                                                                             \
                vkEndCommandBuffer(gCmd);                                                     \
                VkSubmitInfo si = {0};                                                        \
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;                                     \
                si.commandBufferCount = 1;                                                    \
                si.pCommandBuffers = &gCmd;                                                   \
                rc = -6;                                                                      \
                if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) goto done;   \
                if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) goto done;                         \
            } while (0)

        // one submit running a single row-cooperative pipeline over `rows` workgroups
        #define SUBMIT_ROWS(P_, pcPtr, pcSz, rows)                                            \
            do {                                                                              \
                if (ensure_cmd() != 0) { rc = -5; goto done; }                                \
                vkResetCommandBuffer(gCmd, 0);                                               \
                VkCommandBufferBeginInfo cbbi = {0};                                          \
                cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;                     \
                cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;                     \
                vkBeginCommandBuffer(gCmd, &cbbi);                                            \
                vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, (P_)->pipe);          \
                vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, (P_)->plLayout, \
                                        0, 1, &(P_)->dset, 0, NULL);                          \
                vkCmdPushConstants(gCmd, (P_)->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,      \
                                   (pcSz), (pcPtr));                                          \
                vkCmdDispatch(gCmd, (uint32_t)(rows), 1u, 1u);                                \
                vkEndCommandBuffer(gCmd);                                                     \
                VkSubmitInfo si = {0};                                                        \
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;                                     \
                si.commandBufferCount = 1;                                                    \
                si.pCommandBuffers = &gCmd;                                                   \
                rc = -6;                                                                      \
                if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) goto done;   \
                if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) goto done;                         \
            } while (0)

        StridedPC pcs[64];
        if (heads > 64) { rc = -8; goto done; }
        int rep = heads / kvHeads;

        // stage 1: S_h = scale·Q_h·K_hᵀ   (A=Q window, B=K window transB, C=S window)
        {
            VkBuffer b3[3] = { buf[0], buf[1], buf[4] };
            VkDeviceSize l3[3] = { qLen, kLen, sLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                int hh = h / rep;
                StridedPC pc = { seq, dk, seq, 0, 1, dm, dkv, seq, h * dk, hh * dk, h * seq * seq, scale };
                pcs[h] = pc;
            }
            SUBMIT_MATMULS(pcs, heads, gx, gy);
        }
        // stage 2: P = softmax_packed(S)
        {
            VkBuffer b1[1] = { buf[4] };
            VkDeviceSize l1[1] = { sLen };
            dset_write(ps->dset, 1, b1, l1);
            int32_t pc[4] = { heads * seq, seq, seq, causal };
            SUBMIT_ROWS(ps, pc, sizeof(pc), heads * seq);
        }
        // stage 3: dP_h = dO_h·V_hᵀ
        {
            VkBuffer b3[3] = { buf[3], buf[2], buf[5] };
            VkDeviceSize l3[3] = { qLen, kLen, sLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                int hh = h / rep;
                StridedPC pc = { seq, dk, seq, 0, 1, dm, dkv, seq, h * dk, hh * dk, h * seq * seq, 1.0f };
                pcs[h] = pc;
            }
            SUBMIT_MATMULS(pcs, heads, gx, gy);
        }
        // stage 4: dS = scale·P⊙(dP − rowsum(P⊙dP))   (in place in dP)
        {
            VkBuffer b2[2] = { buf[4], buf[5] };
            VkDeviceSize l2[2] = { sLen, sLen };
            dset_write(pj->dset, 2, b2, l2);
            struct { int32_t rows, cols; float alpha; } pc = { heads * seq, seq, scale };
            SUBMIT_ROWS(pj, &pc, sizeof(pc), heads * seq);
        }
        // stage 5: dQ_h = dS_h·K_h
        {
            VkBuffer b3[3] = { buf[5], buf[1], buf[6] };
            VkDeviceSize l3[3] = { sLen, kLen, qLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                int hh = h / rep;
                StridedPC pc = { seq, seq, dk, 0, 0, seq, dkv, dm, h * seq * seq, hh * dk, h * dk, 1.0f };
                pcs[h] = pc;
            }
            SUBMIT_MATMULS(pcs, heads, gxd, gy);
        }
        // stage 6: dVt_h = P_hᵀ·dO_h   (per query head; Go sums the GQA groups)
        {
            VkBuffer b3[3] = { buf[4], buf[3], buf[7] };
            VkDeviceSize l3[3] = { sLen, qLen, tLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                StridedPC pc = { seq, seq, dk, 1, 0, seq, dm, dk, h * seq * seq, h * dk, h * seq * dk, 1.0f };
                pcs[h] = pc;
            }
            SUBMIT_MATMULS(pcs, heads, gxd, gy);
        }
        // stage 7: dKt_h = dS_hᵀ·Q_h   (per query head; Go sums the GQA groups)
        {
            VkBuffer b3[3] = { buf[5], buf[0], buf[8] };
            VkDeviceSize l3[3] = { sLen, qLen, tLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                StridedPC pc = { seq, seq, dk, 1, 0, seq, dm, dk, h * seq * seq, h * dk, h * seq * dk, 1.0f };
                pcs[h] = pc;
            }
            SUBMIT_MATMULS(pcs, heads, gxd, gy);
        }
        #undef SUBMIT_MATMULS
        #undef SUBMIT_ROWS

        download(mem[6], dQ, (size_t)qLen);
        download(mem[7], dVt, (size_t)tLen);
        download(mem[8], dKt, (size_t)tLen);
        rc = 0;
    }
done:
    for (int i = 0; i < 9; ++i) {
        if (idx[i] == -3) continue;
        pool_release(idx[i], buf[i], mem[i]);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// vk_mha_forward_chain (§T531): the matmul-decomposed SDPA FORWARD in the §T528 chain
// structure (three staged submits, ONE descriptor set each; per-head windows via the
// strided offsets): S_h = scale·Q_h·K_hᵀ → P = softmax_packed(S) → O_h = P_h·V_h.
// §T398 rejected an earlier per-head-submit decomposition on the INFERENCE forward;
// §T531's profile showed the forward at 19% of the TRAINING step — this re-measure
// uses the cheaper submit structure (see the §T531 A/B verdict in SPEC).
static int chain_submit_stage(void) {
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gCmd;
    vkEndCommandBuffer(gCmd);
    if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -6;
    if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) return -6;
    return 0;
}

static int chain_begin_stage(void) {
    if (ensure_cmd() != 0) return -5;
    vkResetCommandBuffer(gCmd, 0);
    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(gCmd, &cbbi);
    return 0;
}

int vk_mha_forward_chain(const uint32_t* spvMatmul, int matmulLen,
                         const uint32_t* spvSoftmax, int softmaxLen,
                         const float* Q, const float* K, const float* V, float* O,
                         int seq, int dm, int heads, int kvHeads, int dk,
                         int causal, float scale) {
    if (ensure_init() != 0) return -1;
    pthread_mutex_lock(&gLock);
    int rc = -2;
    int dkv = kvHeads * dk;
    VkDeviceSize qLen = (VkDeviceSize)seq * dm * sizeof(float);
    VkDeviceSize kLen = (VkDeviceSize)seq * dkv * sizeof(float);
    VkDeviceSize sLen = (VkDeviceSize)heads * seq * seq * sizeof(float);
    VkDeviceSize lens[5] = { qLen, kLen, kLen, sLen, qLen }; /* Q K V S/P O */
    VkBuffer buf[5]; VkDeviceMemory mem[5]; int idx[5];
    for (int i = 0; i < 5; ++i) { idx[i] = -3; buf[i] = VK_NULL_HANDLE; mem[i] = VK_NULL_HANDLE; }
    for (int i = 0; i < 5; ++i) {
        idx[i] = pool_acquire(lens[i], &buf[i], &mem[i]);
        if (idx[i] < -1) goto done;
    }
    upload(mem[0], Q, (size_t)qLen);
    upload(mem[1], K, (size_t)kLen);
    upload(mem[2], V, (size_t)kLen);

    {
        PipeCache *pm = NULL, *ps = NULL;
        rc = -4;
        if (pipeline_for(spvMatmul, matmulLen, 3, sizeof(StridedPC), &pm) != 0) goto done;
        if (pipeline_for(spvSoftmax, softmaxLen, 1, 4 * sizeof(int32_t), &ps) != 0) goto done;

        uint32_t gx = ((uint32_t)seq + 15u) / 16u, gy = ((uint32_t)seq + 15u) / 16u;
        uint32_t gxd = ((uint32_t)dk + 15u) / 16u;
        StridedPC pcs[64];
        if (heads > 64) { rc = -8; goto done; }
        int rep = heads / kvHeads;

        /* stage 1: S_h = scale·Q_h·K_hᵀ */
        {
            VkBuffer b3[3] = { buf[0], buf[1], buf[3] };
            VkDeviceSize l3[3] = { qLen, kLen, sLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                int hh = h / rep;
                StridedPC pc = { seq, dk, seq, 0, 1, dm, dkv, seq, h * dk, hh * dk, h * seq * seq, scale };
                pcs[h] = pc;
            }
            if ((rc = chain_begin_stage()) != 0) goto done;
            vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->pipe);
            vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->plLayout, 0, 1, &pm->dset, 0, NULL);
            for (int h = 0; h < heads; ++h) {
                vkCmdPushConstants(gCmd, pm->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(StridedPC), &pcs[h]);
                vkCmdDispatch(gCmd, gx, gy, 1u);
            }
            if ((rc = chain_submit_stage()) != 0) goto done;
        }
        /* stage 2: P = softmax_packed(S) */
        {
            VkBuffer b1[1] = { buf[3] };
            VkDeviceSize l1[1] = { sLen };
            dset_write(ps->dset, 1, b1, l1);
            int32_t pc[4] = { heads * seq, seq, seq, causal };
            if ((rc = chain_begin_stage()) != 0) goto done;
            vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipe);
            vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->plLayout, 0, 1, &ps->dset, 0, NULL);
            vkCmdPushConstants(gCmd, ps->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(gCmd, (uint32_t)(heads * seq), 1u, 1u);
            if ((rc = chain_submit_stage()) != 0) goto done;
        }
        /* stage 3: O_h = P_h·V_h */
        {
            VkBuffer b3[3] = { buf[3], buf[2], buf[4] };
            VkDeviceSize l3[3] = { sLen, kLen, qLen };
            dset_write(pm->dset, 3, b3, l3);
            for (int h = 0; h < heads; ++h) {
                int hh = h / rep;
                StridedPC pc = { seq, seq, dk, 0, 0, seq, dkv, dm, h * seq * seq, hh * dk, h * dk, 1.0f };
                pcs[h] = pc;
            }
            if ((rc = chain_begin_stage()) != 0) goto done;
            vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->pipe);
            vkCmdBindDescriptorSets(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pm->plLayout, 0, 1, &pm->dset, 0, NULL);
            for (int h = 0; h < heads; ++h) {
                vkCmdPushConstants(gCmd, pm->plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(StridedPC), &pcs[h]);
                vkCmdDispatch(gCmd, gxd, gy, 1u);
            }
            if ((rc = chain_submit_stage()) != 0) goto done;
        }
        download(mem[4], O, (size_t)qLen);
        rc = 0;
    }
done:
    for (int i = 0; i < 5; ++i) {
        if (idx[i] == -3) continue;
        pool_release(idx[i], buf[i], mem[i]);
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}
