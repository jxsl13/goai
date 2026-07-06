//go:build vulkan && cgo

// Vulkan compute side of the vulkan backend (§T43). Compiled only under
// `-tags vulkan` with the Vulkan loader. A process-wide instance/device/queue is
// created lazily; each matmul creates its buffers, descriptor set, pipeline, and
// command buffer, dispatches, waits, and frees. Verbose but the standard Vulkan
// compute sequence — confirmed vs the Vulkan spec + compute tutorials (§R44).

#include "vk_bridge.h"
#include <vulkan/vulkan.h>
#include <string.h>
#include <stdlib.h>

static VkInstance       gInstance = VK_NULL_HANDLE;
static VkPhysicalDevice gPhys     = VK_NULL_HANDLE;
static VkDevice         gDevice   = VK_NULL_HANDLE;
static VkQueue          gQueue    = VK_NULL_HANDLE;
static uint32_t         gQueueFamily = 0;
static int              gInitTried = 0;
static int              gInitOK    = 0;

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

static int ensure_init(void) {
    if (gInitTried) return gInitOK ? 0 : -1;
    gInitTried = 1;

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "goai";
    app.apiVersion = VK_API_VERSION_1_1;

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

    int qf = -1;
    for (uint32_t i = 0; i < nd; ++i) {
        int q = find_compute_queue(devs[i]);
        if (q >= 0) { gPhys = devs[i]; qf = q; break; }
    }
    free(devs);
    if (qf < 0) return -1;
    gQueueFamily = (uint32_t)qf;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = gQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // A portability physical device (MoltenVK) requires enabling
    // VK_KHR_portability_subset at logical-device creation. Absent elsewhere.
    const char* devExts[1];
    uint32_t nDevExts = 0;
    if (has_device_ext(gPhys, "VK_KHR_portability_subset")) {
        devExts[nDevExts++] = "VK_KHR_portability_subset";
    }

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nDevExts;
    dci.ppEnabledExtensionNames = nDevExts ? devExts : NULL;
    if (vkCreateDevice(gPhys, &dci, NULL, &gDevice) != VK_SUCCESS) return -1;

    vkGetDeviceQueue(gDevice, gQueueFamily, 0, &gQueue);
    gInitOK = 1;
    return 0;
}

int vk_available(void) { return ensure_init() == 0 ? 1 : 0; }

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

int vk_matmul_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* C,
                  int M, int K, int N) {
    if (ensure_init() != 0) return -1;

    VkDeviceSize aLen = (VkDeviceSize)M * K * sizeof(float);
    VkDeviceSize bLen = (VkDeviceSize)K * N * sizeof(float);
    VkDeviceSize cLen = (VkDeviceSize)M * N * sizeof(float);

    VkBuffer aBuf = VK_NULL_HANDLE, bBuf = VK_NULL_HANDLE, cBuf = VK_NULL_HANDLE;
    VkDeviceMemory aMem = VK_NULL_HANDLE, bMem = VK_NULL_HANDLE, cMem = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout plLayout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    int rc = -2;

    if (make_buffer(aLen, &aBuf, &aMem) != 0) goto done;
    if (make_buffer(bLen, &bBuf, &bMem) != 0) goto done;
    if (make_buffer(cLen, &cBuf, &cMem) != 0) goto done;
    upload(aMem, A, (size_t)aLen);
    upload(bMem, B, (size_t)bLen);

    // Descriptor set layout: 3 storage buffers at bindings 0,1,2 (compute stage).
    VkDescriptorSetLayoutBinding binds[3];
    for (int i = 0; i < 3; ++i) {
        binds[i] = (VkDescriptorSetLayoutBinding){0};
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(gDevice, &dslci, NULL, &dsl) != VK_SUCCESS) { rc = -3; goto done; }

    VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    if (vkCreateDescriptorPool(gDevice, &dpci, NULL, &pool) != VK_SUCCESS) { rc = -3; goto done; }

    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    if (vkAllocateDescriptorSets(gDevice, &dsai, &dset) != VK_SUCCESS) { rc = -3; goto done; }

    VkDescriptorBufferInfo bufInfo[3] = {
        { aBuf, 0, aLen }, { bBuf, 0, bLen }, { cBuf, 0, cLen },
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; ++i) {
        writes[i] = (VkWriteDescriptorSet){0};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfo[i];
    }
    vkUpdateDescriptorSets(gDevice, 3, writes, 0, NULL);

    // Push constants: M, K, N.
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(int32_t) };

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = (size_t)spvLen;
    smci.pCode = spv;
    if (vkCreateShaderModule(gDevice, &smci, NULL, &shader) != VK_SUCCESS) { rc = -4; goto done; }

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(gDevice, &plci, NULL, &plLayout) != VK_SUCCESS) { rc = -4; goto done; }

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = plLayout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    if (vkCreateComputePipelines(gDevice, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS) { rc = -4; goto done; }

    // Command buffer: bind, push dims, dispatch ceil(N/16) x ceil(M/16).
    VkCommandPoolCreateInfo cpc = {0};
    cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.queueFamilyIndex = gQueueFamily;
    if (vkCreateCommandPool(gDevice, &cpc, NULL, &cmdPool) != VK_SUCCESS) { rc = -5; goto done; }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(gDevice, &cbai, &cmd) != VK_SUCCESS) { rc = -5; goto done; }

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plLayout, 0, 1, &dset, 0, NULL);
    int32_t dims[3] = { M, K, N };
    vkCmdPushConstants(cmd, plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dims), dims);
    uint32_t gx = ((uint32_t)N + 15u) / 16u;
    uint32_t gy = ((uint32_t)M + 15u) / 16u;
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) { rc = -6; goto done; }
    if (vkQueueWaitIdle(gQueue) != VK_SUCCESS) { rc = -6; goto done; }

    download(cMem, C, (size_t)cLen);
    rc = 0;

done:
    if (cmdPool)  vkDestroyCommandPool(gDevice, cmdPool, NULL);
    if (pipe)     vkDestroyPipeline(gDevice, pipe, NULL);
    if (plLayout) vkDestroyPipelineLayout(gDevice, plLayout, NULL);
    if (shader)   vkDestroyShaderModule(gDevice, shader, NULL);
    if (pool)     vkDestroyDescriptorPool(gDevice, pool, NULL);
    if (dsl)      vkDestroyDescriptorSetLayout(gDevice, dsl, NULL);
    if (aBuf)     vkDestroyBuffer(gDevice, aBuf, NULL);
    if (bBuf)     vkDestroyBuffer(gDevice, bBuf, NULL);
    if (cBuf)     vkDestroyBuffer(gDevice, cBuf, NULL);
    if (aMem)     vkFreeMemory(gDevice, aMem, NULL);
    if (bMem)     vkFreeMemory(gDevice, bMem, NULL);
    if (cMem)     vkFreeMemory(gDevice, cMem, NULL);
    return rc;
}
