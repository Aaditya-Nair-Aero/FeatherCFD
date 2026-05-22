import importlib, sys, os
_site_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'venv', 'lib', 'python3.12', 'site-packages')
_site_pkg = os.path.abspath(_site_pkg)
if os.path.isdir(_site_pkg) and _site_pkg not in sys.path:
    sys.path.insert(0, _site_pkg)
importlib.import_module('vulkan._vulkancache')
importlib.import_module('vulkan._vulkan')
import vulkan as vk
import numpy as np
from .buffer import VKBuffer

class VKContext:
    def __init__(self, enable_validation=False, headless=True):
        self.headless = headless
        self.instance = None
        self.physical_device = None
        self.device = None
        self.compute_queue = None
        self.compute_family = None
        self.graphics_queue = None
        self.graphics_family = None
        self.transfer_queue = None
        self.transfer_family = None
        self.memory_properties = None
        self.command_pool = None
        self._create_instance(enable_validation)
        self._pick_physical_device()
        self._create_device()
        self._create_command_pool()

    def _create_instance(self, enable_validation):
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GPU CFD Solver",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="CFD Vulkan",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        layers = ["VK_LAYER_KHRONOS_validation"] if enable_validation else []
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers if layers else None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
        )
        self.instance = vk.vkCreateInstance(create_info, None)
        if self.instance is None:
            raise RuntimeError("Failed to create Vulkan instance")

    def _pick_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan-capable GPU found")
        for dev in devices:
            props = vk.vkGetPhysicalDeviceProperties(dev)
            if props.deviceType in (vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                                     vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU):
                self.physical_device = dev
                self.device_name = props.deviceName
                break
        if self.physical_device is None:
            self.physical_device = devices[0]
            props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
            self.device_name = props.deviceName
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        print(f"  Vulkan GPU: {self.device_name}")

    def _create_device(self):
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)

        for i, family in enumerate(families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT and family.queueCount > 0:
                if self.compute_family is None:
                    self.compute_family = i
            if family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT and family.queueCount > 0:
                if self.graphics_family is None:
                    self.graphics_family = i
            if family.queueFlags & vk.VK_QUEUE_TRANSFER_BIT and family.queueCount > 0:
                if self.transfer_family is None:
                    self.transfer_family = i

        if self.compute_family is None:
            self.compute_family = 0
        if self.transfer_family is None:
            self.transfer_family = self.compute_family

        unique_families = set()
        if not self.headless and self.graphics_family is not None:
            unique_families.add(self.graphics_family)
        unique_families.add(self.compute_family)
        family_priorities = [1.0]

        queue_infos = []
        for qf in unique_families:
            queue_infos.append(vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=qf,
                queueCount=1,
                pQueuePriorities=family_priorities,
            ))

        device_features = vk.VkPhysicalDeviceFeatures(
            shaderStorageImageExtendedFormats=vk.VK_TRUE,
            shaderInt64=vk.VK_TRUE,
        )

        create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=queue_infos,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            pEnabledFeatures=device_features,
        )
        self.device = vk.vkCreateDevice(self.physical_device, create_info, None)

        self.compute_queue = vk.vkGetDeviceQueue(self.device, self.compute_family, 0)
        if not self.headless and self.graphics_family is not None:
            self.graphics_queue = vk.vkGetDeviceQueue(self.device, self.graphics_family, 0)
        self.transfer_queue = vk.vkGetDeviceQueue(self.device, self.transfer_family, 0)

    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.compute_family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

    def create_command_buffer(self):
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

    def find_memory_type(self, type_filter, properties):
        for i in range(self.memory_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (self.memory_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")

    def create_buffer(self, size, usage, memory_properties):
        return VKBuffer(self, size, usage, memory_properties)

    def submit_and_wait(self, command_buffer, fence=None):
        if fence is None:
            fence = vk.vkCreateFence(self.device,
                vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
            own_fence = True
        else:
            own_fence = False

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10 ** 9)
        if own_fence:
            vk.vkDestroyFence(self.device, fence, None)

    def begin_command_buffer(self, cmd):
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

    def end_and_submit(self, cmd):
        vk.vkEndCommandBuffer(cmd)
        self.submit_and_wait(cmd)

    def destroy(self):
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)
