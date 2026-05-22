"""
Vulkan-based volumetric raymarching visualizer for pre-computed CFD data.

Streams frames from NVME SSD to GPU per-frame.
Usage:
    CFD_DATA_DIR=/path/to/frames python -m vk_cfd.viz.visualize_vk
"""

import glfw
import vulkan as vk
import numpy as np
import ctypes
import math
import os
import glob
import sys
import struct
from scipy.ndimage import distance_transform_edt

from ..common.shader import compile_file_to_spirv, create_shader_module
from ..common.buffer import VKBuffer
from vulkan._vulkan import ffi

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shaders')


# ── Helpers ───────────────────────────────────────────────────────────
def c_void(val=0):
    return ctypes.c_void_p(val)


def make_image(ctx, device, fmt, size, usage):
    extent = vk.VkExtent3D(width=size, height=size, depth=size)
    img_info = vk.VkImageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=vk.VK_IMAGE_TYPE_3D,
        format=fmt, extent=extent,
        mipLevels=1, arrayLayers=1,
        samples=vk.VK_SAMPLE_COUNT_1_BIT,
        tiling=vk.VK_IMAGE_TILING_OPTIMAL,
        usage=usage,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
    )
    img = vk.vkCreateImage(device, img_info, None)
    reqs = vk.vkGetImageMemoryRequirements(device, img)
    mem_type = ctx.find_memory_type(reqs.memoryTypeBits,
                                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    alloc = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=reqs.size,
        memoryTypeIndex=mem_type,
    )
    mem = vk.vkAllocateMemory(device, alloc, None)
    vk.vkBindImageMemory(device, img, mem, 0)
    return img, mem


def make_view(device, img, fmt):
    view_info = vk.VkImageViewCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=img,
        viewType=vk.VK_IMAGE_VIEW_TYPE_3D,
        format=fmt,
        components=vk.VkComponentMapping(
            r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        ),
        subresourceRange=vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        ),
    )
    return vk.vkCreateImageView(device, view_info, None)


def transition_layout(cmd, device, img, old, new, src_stage, dst_stage,
                      src_access=0, dst_access=0):
    barrier = vk.VkImageMemoryBarrier(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        srcAccessMask=src_access,
        dstAccessMask=dst_access,
        oldLayout=old, newLayout=new,
        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
        image=img,
        subresourceRange=vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        ),
    )
    vk.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, None, 0, None, 1, [barrier])


class VKVisualizer:
    def __init__(self):
        data_dir = os.environ.get('CFD_DATA_DIR', '/home/aaditya/Downloads/tmp')
        self.data_dir = data_dir
        self.frame_files = sorted(glob.glob(os.path.join(data_dir, 'vel_*.npy')))
        if not self.frame_files:
            raise FileNotFoundError(
                f"No simulation data found in {data_dir}/")
        first = np.load(self.frame_files[0])
        self.grid_size = first.shape[0]
        print(f"Frames: {len(self.frame_files)}, grid: {self.grid_size}³")

        self.frame_idx = 0
        self.playing = True
        self.playback_speed = 15.0
        self.time_accum = 0.0
        self.viz_mode = 0
        self.last_time = time.time()

        self.cam_dist = 3.8
        self.cam_yaw = -0.5
        self.cam_pitch = 0.3
        self.dragging = False
        self.mouse_x = self.mouse_y = 0

        self._init_vulkan()
        self._create_swapchain()
        self._create_render_pass()
        self._create_pipeline()
        self._create_textures()
        self._create_descriptors()
        self._create_framebuffers()
        self._create_cmd_buffers()
        self._create_sync_objects()
        self._upload_frame(0)
        print("Ready! Mouse: orbit | Scroll: zoom | Space: pause | V: cycle mode")

    # ── Vulkan Init ──────────────────────────────────────────────────
    def _init_vulkan(self):
        glfw.init()
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self.window = glfw.create_window(1280, 720, "CFD Vulkan Visualizer", None, None)
        if not self.window:
            raise RuntimeError("Failed to create GLFW window")

        exts = glfw.get_required_instance_extensions()
        ext_names = list(exts)
        layer_names = []

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="CFD Vulkan Visualizer",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="CFD Vulkan",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        inst_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(ext_names),
            ppEnabledExtensionNames=ext_names,
            enabledLayerCount=len(layer_names),
            ppEnabledLayerNames=layer_names if layer_names else None,
        )
        self.instance = vk.vkCreateInstance(inst_info, None)
        self._setup_debug()

        # Pick GPU
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        self.phys_dev = devices[0]
        props = vk.vkGetPhysicalDeviceProperties(self.phys_dev)
        print(f"GPU: {props.deviceName}")

        self.memory_props = vk.vkGetPhysicalDeviceMemoryProperties(self.phys_dev)
        self._load_khr_functions()

        # Create surface FIRST, then find queue families
        surf_ptr = ctypes.c_void_p()
        glfw.create_window_surface(self.instance, self.window, None, ctypes.byref(surf_ptr))
        self.surface = surf_ptr.value
        self._find_queue_families()

        # Device with swapchain + graphics
        device_features = vk.VkPhysicalDeviceFeatures(
            shaderStorageImageExtendedFormats=vk.VK_TRUE,
        )
        priorities = [1.0]
        q_infos = []
        families = set()
        families.add(self.graphics_family)
        if self.present_family != self.graphics_family:
            families.add(self.present_family)
        for qf in families:
            q_infos.append(vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=qf, queueCount=1,
                pQueuePriorities=priorities,
            ))

        dev_exts = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        dev_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(q_infos),
            pQueueCreateInfos=q_infos,
            enabledExtensionCount=len(dev_exts),
            ppEnabledExtensionNames=dev_exts,
            pEnabledFeatures=device_features,
        )
        self.device = vk.vkCreateDevice(self.phys_dev, dev_info, None)

        self.graphics_queue = vk.vkGetDeviceQueue(self.device, self.graphics_family, 0)
        self.present_queue = vk.vkGetDeviceQueue(self.device, self.present_family, 0)

        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.graphics_family,
        )
        self.cmd_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

    def _setup_debug(self):
        pass

    def _load_khr_functions(self):
        def load(name):
            return vk.vkGetInstanceProcAddr(self.instance, name)
        self.khr_surface_support = load('vkGetPhysicalDeviceSurfaceSupportKHR')
        self.khr_surface_caps = load('vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        self.khr_surface_formats = load('vkGetPhysicalDeviceSurfaceFormatsKHR')
        self.khr_surface_present_modes = load('vkGetPhysicalDeviceSurfacePresentModesKHR')
        self.khr_destroy_surface = load('vkDestroySurfaceKHR')
        self.khr_create_swapchain = load('vkCreateSwapchainKHR')
        self.khr_get_swapchain_images = load('vkGetSwapchainImagesKHR')
        self.khr_destroy_swapchain = load('vkDestroySwapchainKHR')
        self.khr_acquire_next_image = load('vkAcquireNextImageKHR')
        self.khr_queue_present = load('vkQueuePresentKHR')

    def _find_queue_families(self):
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.phys_dev)
        self.graphics_family = None
        self.present_family = None
        for i, f in enumerate(families):
            if f.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                self.graphics_family = i
        for i, f in enumerate(families):
            supported = self.khr_surface_support(
                self.phys_dev, i, self.surface)
            if supported:
                self.present_family = i
                break
        if self.present_family is None:
            self.present_family = self.graphics_family

    def find_memory_type(self, type_filter, props):
        for i in range(self.memory_props.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (self.memory_props.memoryTypes[i].propertyFlags & props) == props:
                return i
        raise RuntimeError("No suitable memory type")

    # ── Swapchain ────────────────────────────────────────────────────
    def _create_swapchain(self):
        caps = self.khr_surface_caps(self.phys_dev, self.surface)
        formats = self.khr_surface_formats(self.phys_dev, self.surface)
        self.swap_fmt = formats[0].format if len(formats) > 0 else vk.VK_FORMAT_B8G8R8A8_UNORM

        present_modes = self.khr_surface_present_modes(self.phys_dev, self.surface)
        self.present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            self.present_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR

        extent = caps.currentExtent
        if extent.width == 0xFFFFFFFF:
            extent = vk.VkExtent2D(width=1280, height=720)
        self.swap_extent = extent

        min_images = max(caps.minImageCount, 2)
        swap_info = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface, minImageCount=min_images,
            imageFormat=self.swap_fmt, imageColorSpace=formats[0].colorSpace,
            imageExtent=extent, imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=self.present_mode, clipped=vk.VK_TRUE,
        )
        self.swapchain = self.khr_create_swapchain(self.device, swap_info, None)

        self.swap_images = self.khr_get_swapchain_images(self.device, self.swapchain)
        self.swap_views = []
        for img in self.swap_images:
            self.swap_views.append(make_view(self.device, img, self.swap_fmt))

    def _recreate_swapchain(self):
        vk.vkDeviceWaitIdle(self.device)
        for fb in self.framebuffers:
            vk.vkDestroyFramebuffer(self.device, fb, None)
        for sv in self.swap_views:
            vk.vkDestroyImageView(self.device, sv, None)
        self.khr_destroy_swapchain(self.device, self.swapchain, None)
        self._create_swapchain()
        self._create_framebuffers()

    # ── Render Pass ──────────────────────────────────────────────────
    def _create_render_pass(self):
        color_attach = vk.VkAttachmentDescription(
            format=self.swap_fmt,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1, pColorAttachments=[color_ref],
        )
        dep = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )
        rp_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1, pAttachments=[color_attach],
            subpassCount=1, pSubpasses=[subpass],
            dependencyCount=1, pDependencies=[dep],
        )
        self.render_pass = vk.vkCreateRenderPass(self.device, rp_info, None)

    # ── Graphics Pipeline ────────────────────────────────────────────
    def _create_pipeline(self):
        vert_path = os.path.join(SHADER_DIR, 'raymarch_vk.vert')
        frag_path = os.path.join(SHADER_DIR, 'raymarch_vk.frag')
        vert_spv = compile_file_to_spirv(vert_path, 'vertex')
        frag_spv = compile_file_to_spirv(frag_path, 'fragment')
        vert_mod = create_shader_module(self.device, vert_spv)
        frag_mod = create_shader_module(self.device, frag_spv)

        vert_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vert_mod,
            pName="main",
        )
        frag_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=frag_mod,
            pName="main",
        )

        # Push constant range (cam_pos + pad + mat4)
        pc_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            offset=0, size=4 + 12 + 64,  # float pad + vec3 + mat4 = 80 bytes
        )

        # Descriptor set layout (binding 0-3)
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=3, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ),
        ]
        dsl_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings), pBindings=bindings,
        )
        self.desc_set_layout = vk.vkCreateDescriptorSetLayout(self.device, dsl_info, None)

        # Pipeline layout
        pl_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1, pSetLayouts=[self.desc_set_layout],
            pushConstantRangeCount=1, pPushConstantRanges=[pc_range],
        )
        pipe_layout = vk.vkCreatePipelineLayout(self.device, pl_info, None)

        # Vertex input (no vertex buffers)
        vi_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0,
        )
        ia_info = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        )
        viewport = vk.VkViewport(
            x=0, y=0,
            width=self.swap_extent.width,
            height=self.swap_extent.height,
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=self.swap_extent,
        )
        vs_info = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1, pViewports=[viewport],
            scissorCount=1, pScissors=[scissor],
        )
        rs_info = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            lineWidth=1.0,
        )
        ms_info = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
        )
        cb_info = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT |
                            vk.VK_COLOR_COMPONENT_G_BIT |
                            vk.VK_COLOR_COMPONENT_B_BIT |
                            vk.VK_COLOR_COMPONENT_A_BIT),
            blendEnable=vk.VK_FALSE,
        )
        cb_state = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount=1, pAttachments=[cb_info],
        )

        pipe_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2, pStages=[vert_stage, frag_stage],
            pVertexInputState=vi_info,
            pInputAssemblyState=ia_info,
            pViewportState=vs_info,
            pRasterizationState=rs_info,
            pMultisampleState=ms_info,
            pColorBlendState=cb_state,
            layout=pipe_layout,
            renderPass=self.render_pass, subpass=0,
        )
        self.pipeline = vk.vkCreateGraphicsPipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [pipe_info], None)[0]
        self.pipeline_layout = pipe_layout

        vk.vkDestroyShaderModule(self.device, vert_mod, None)
        vk.vkDestroyShaderModule(self.device, frag_mod, None)

    # ── Textures ─────────────────────────────────────────────────────
    def _create_textures(self):
        vol_fmt = vk.VK_FORMAT_R16G16B16A16_SFLOAT
        obs_fmt = vk.VK_FORMAT_R8_UINT
        sdf_fmt = vk.VK_FORMAT_R16_SFLOAT
        usage = vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT

        def make(fmt):
            img, mem = make_image(self, self.device, fmt, self.grid_size, usage)
            view = make_view(self.device, img, fmt)
            return img, view, mem

        self.vol_img, self.vol_view, self.vol_mem = make(vol_fmt)
        self.obs_img, self.obs_view, self.obs_mem = make(obs_fmt)
        self.sdf_img, self.sdf_view, self.sdf_mem = make(sdf_fmt)

        # Sampler
        sampler_info = vk.VkSamplerCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter=vk.VK_FILTER_LINEAR,
            minFilter=vk.VK_FILTER_LINEAR,
            addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
        )
        self.sampler = vk.vkCreateSampler(self.device, sampler_info, None)

        # Uniform buffer for params
        self.params_buf = VKBuffer(self, 32,
                                   vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

    # ── Descriptors ──────────────────────────────────────────────────
    def _create_descriptors(self):
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=16,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=4,
            ),
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=4, poolSizeCount=len(pool_sizes), pPoolSizes=pool_sizes,
        )
        self.desc_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.desc_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.desc_set_layout],
        )
        self.desc_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        # Write descriptors
        vol_info = vk.VkDescriptorImageInfo(
            sampler=self.sampler,
            imageView=self.vol_view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        obs_info = vk.VkDescriptorImageInfo(
            sampler=self.sampler,
            imageView=self.obs_view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        sdf_info = vk.VkDescriptorImageInfo(
            sampler=self.sampler,
            imageView=self.sdf_view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        buf_info = vk.VkDescriptorBufferInfo(
            buffer=self.params_buf.buffer,
            offset=0, range=32,
        )

        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[vol_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=1,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[obs_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=2,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[sdf_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=3,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=[buf_info],
            ),
        ]
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

    # ── Framebuffers ─────────────────────────────────────────────────
    def _create_framebuffers(self):
        self.framebuffers = []
        for sv in self.swap_views:
            fb_info = vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.render_pass,
                attachmentCount=1, pAttachments=[sv],
                width=self.swap_extent.width,
                height=self.swap_extent.height,
                layers=1,
            )
            self.framebuffers.append(
                vk.vkCreateFramebuffer(self.device, fb_info, None))

    # ── Command Buffers ──────────────────────────────────────────────
    def _create_cmd_buffers(self):
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.cmd_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.framebuffers),
        )
        self.cmd_buffers = vk.vkAllocateCommandBuffers(self.device, alloc_info)

    # ── Sync Objects ─────────────────────────────────────────────────
    def _create_sync_objects(self):
        sem_info = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        self.img_avail = vk.vkCreateSemaphore(self.device, sem_info, None)
        self.render_done = vk.vkCreateSemaphore(self.device, sem_info, None)

        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
        )
        self.in_flight = vk.vkCreateFence(self.device, fence_info, None)

    # ── Frame Upload ─────────────────────────────────────────────────
    def _upload_frame(self, idx):
        data = np.load(self.frame_files[idx])

        def upload_to_image(img, arr, fmt, components):
            np_dtype = np.float16 if fmt in (vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                                              vk.VK_FORMAT_R16_SFLOAT) else np.uint8
            expected_shape = (self.grid_size,) * 3 if components == 1 else (self.grid_size,) * 3 + (components,)
            arr = np.ascontiguousarray(arr, dtype=np_dtype).reshape(expected_shape)
            byte_size = arr.nbytes

            staging = VKBuffer(self, byte_size,
                               vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            staging.upload(arr.tobytes())

            cmd_alloc = vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            )
            cmd = vk.vkAllocateCommandBuffers(self.device, cmd_alloc)[0]
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            )
            vk.vkBeginCommandBuffer(cmd, begin_info)

            transition_layout(cmd, self.device, img,
                              vk.VK_IMAGE_LAYOUT_UNDEFINED,
                              vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, vk.VK_ACCESS_TRANSFER_WRITE_BIT)

            region = vk.VkBufferImageCopy(
                bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0, baseArrayLayer=0, layerCount=1,
                ),
                imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                imageExtent=vk.VkExtent3D(
                    width=self.grid_size, height=self.grid_size, depth=self.grid_size),
            )
            vk.vkCmdCopyBufferToImage(cmd, staging.buffer, img,
                                      vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      1, [region])

            transition_layout(cmd, self.device, img,
                              vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                              vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                              vk.VK_ACCESS_SHADER_READ_BIT)

            vk.vkEndCommandBuffer(cmd)
            submit = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1, pCommandBuffers=[cmd],
            )
            fence = vk.vkCreateFence(self.device,
                vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
            vk.vkQueueSubmit(self.graphics_queue, 1, [submit], fence)
            vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10**9)
            vk.vkDestroyFence(self.device, fence, None)
            vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])
            staging.destroy()

        # Upload volume (RGBA16F, 4 components)
        upload_to_image(self.vol_img, data, vk.VK_FORMAT_R16G16B16A16_SFLOAT, 4)

        # Upload obstacle + SDF if available
        obs_path = os.path.join(self.data_dir, 'obstacles.npy')
        if os.path.exists(obs_path) and idx == 0:
            obs_arr = np.load(obs_path)
            upload_to_image(self.obs_img, obs_arr, vk.VK_FORMAT_R8_UINT, 1)

            dist_out = distance_transform_edt(1 - obs_arr).astype(np.float16)
            dist_in = distance_transform_edt(obs_arr).astype(np.float16)
            sdf_arr = (dist_out - dist_in).astype(np.float16)
            upload_to_image(self.sdf_img, sdf_arr, vk.VK_FORMAT_R16_SFLOAT, 1)

        self.frame_idx = idx

    # ── Camera ───────────────────────────────────────────────────────
    def _cam_pos(self):
        x = self.cam_dist * math.cos(self.cam_pitch) * math.sin(self.cam_yaw)
        y = self.cam_dist * math.sin(self.cam_pitch)
        z = self.cam_dist * math.cos(self.cam_pitch) * math.cos(self.cam_yaw)
        return np.array([x, y, z], dtype='f4')

    @staticmethod
    def _look_at(eye, target, up):
        f = target - eye
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-8:
            return np.eye(4, dtype='f4')
        f = f / f_norm
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-6:
            s = np.array([1.0, 0.0, 0.0], dtype='f4')
        else:
            s = s / s_norm
        u = np.cross(s, f)
        M = np.eye(4, dtype='f4')
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype='f4')
        T[:3, 3] = -eye
        return M @ T

    @staticmethod
    def _perspective(fov_deg, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        M = np.zeros((4, 4), dtype='f4')
        M[0, 0] = f / aspect
        M[1, 1] = f
        M[2, 2] = (far + near) / (near - far)
        M[2, 3] = (2 * far * near) / (near - far)
        M[3, 2] = -1.0
        return M

    # ── Record Command Buffer ────────────────────────────────────────
    def _record_cmd(self, fb_idx):
        cmd = self.cmd_buffers[fb_idx]
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        clear = vk.VkClearValue(color=vk.VkClearColorValue(
            float32=[0.02, 0.02, 0.02, 1.0]))
        rp_begin = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self.render_pass,
            framebuffer=self.framebuffers[fb_idx],
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=self.swap_extent,
            ),
            clearValueCount=1, pClearValues=[clear],
        )
        vk.vkCmdBeginRenderPass(cmd, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   self.pipeline_layout, 0, 1, [self.desc_set], 0, None)

        # Push constants: cam_pos + pad + inv_view_proj
        eye = self._cam_pos()
        w, h = self.swap_extent.width, self.swap_extent.height
        aspect = w / h if h > 0 else 1.0
        target = np.array([0.0, 0.0, 0.0], dtype='f4')
        up = np.array([0.0, 1.0, 0.0], dtype='f4')
        view = self._look_at(eye, target, up)
        proj = self._perspective(55.0, aspect, 0.1, 100.0)
        inv_vp = np.linalg.inv(proj @ view)
        pc_data = struct.pack('3f', *eye) + struct.pack('f', 0.0) + inv_vp.T.astype('f4').tobytes()
        pc_buf = ffi.new('char[' + str(len(pc_data)) + ']')
        ffi.memmove(pc_buf, pc_data, len(pc_data))
        vk.vkCmdPushConstants(cmd, self.pipeline_layout,
                              vk.VK_SHADER_STAGE_VERTEX_BIT, 0,
                              len(pc_data), pc_buf)

        # Update params uniform buffer
        params_data = struct.pack('i', self.viz_mode)  # viz_mode
        params_data += struct.pack('f', 35.0)  # density_scale
        params_data += struct.pack('f', 0.006)  # step_size
        params_data += struct.pack('i', 160)    # num_steps
        params_data += struct.pack('i', self.grid_size)  # grid_size
        params_data += struct.pack('12s', b'\x00' * 12)  # pad
        self.params_buf.upload(params_data)

        vk.vkCmdDraw(cmd, 6, 1, 0, 0)
        vk.vkCmdEndRenderPass(cmd)
        vk.vkEndCommandBuffer(cmd)

    # ── Draw Frame ───────────────────────────────────────────────────
    def draw_frame(self):
        vk.vkWaitForFences(self.device, 1, [self.in_flight], vk.VK_TRUE, 10**9)

        img_idx = self.khr_acquire_next_image(self.device, self.swapchain,
                                                10**9, self.img_avail,
                                                vk.VK_NULL_HANDLE)
        if img_idx == vk.VK_ERROR_OUT_OF_DATE_KHR or img_idx == vk.VK_SUBOPTIMAL_KHR:
            self._recreate_swapchain()
            return

        vk.vkResetFences(self.device, 1, [self.in_flight])
        self._record_cmd(img_idx)

        wait_sems = [self.img_avail]
        wait_stages = [vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        signal_sems = [self.render_done]

        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1, pWaitSemaphores=wait_sems,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1,
            pCommandBuffers=[self.cmd_buffers[img_idx]],
            signalSemaphoreCount=1, pSignalSemaphores=signal_sems,
        )
        vk.vkQueueSubmit(self.graphics_queue, 1, [submit], self.in_flight)

        present = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1, pWaitSemaphores=signal_sems,
            swapchainCount=1, pSwapchains=[self.swapchain],
            pImageIndices=[img_idx],
        )
        result = self.khr_queue_present(self.present_queue, present)
        if result in (vk.VK_ERROR_OUT_OF_DATE_KHR, vk.VK_SUBOPTIMAL_KHR):
            self._recreate_swapchain()

    # ── Main Loop ────────────────────────────────────────────────────
    def run(self):
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_scroll_callback(self.window, self._scroll)
        glfw.set_key_callback(self.window, self._key)
        glfw.set_framebuffer_size_callback(self.window, self._resize)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            if self.playing and len(self.frame_files) > 1:
                self.time_accum += dt
                frames_to_advance = int(self.time_accum * self.playback_speed)
                if frames_to_advance > 0:
                    self.time_accum -= frames_to_advance / self.playback_speed
                    n = len(self.frame_files)
                    self.frame_idx = (self.frame_idx + frames_to_advance) % n
                    self._upload_frame(self.frame_idx)

            self.draw_frame()

        vk.vkDeviceWaitIdle(self.device)
        self._cleanup()

    def _mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS

    def _mouse_move(self, win, x, y):
        if self.dragging:
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            self.cam_yaw += dx * 0.005
            self.cam_pitch += dy * 0.005
            self.cam_pitch = max(-1.2, min(1.2, self.cam_pitch))
        self.mouse_x, self.mouse_y = x, y

    def _scroll(self, win, xoff, yoff):
        self.cam_dist += yoff * 0.25
        self.cam_dist = max(1.5, min(12.0, self.cam_dist))

    def _key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            self.playing = not self.playing
            state = ">> Playing" if self.playing else "|| Paused"
            print(f"[{state}] Frame {self.frame_idx}/{len(self.frame_files)-1}")
        elif key == glfw.KEY_V:
            self.viz_mode = (self.viz_mode + 1) % 5
            modes = ["Velocity", "Dye", "Vorticity", "Near Object", "Velocity (SDF)"]
            if self.viz_mode >= len(modes):
                self.viz_mode = 0
            print(f"Viz Mode: {modes[self.viz_mode]}")
        elif key == glfw.KEY_RIGHT:
            self.frame_idx = (self.frame_idx + 1) % len(self.frame_files)
            self._upload_frame(self.frame_idx)
        elif key == glfw.KEY_LEFT:
            self.frame_idx = (self.frame_idx - 1) % len(self.frame_files)
            self._upload_frame(self.frame_idx)

    def _resize(self, win, w, h):
        if w > 0 and h > 0:
            self._recreate_swapchain()

    def _cleanup(self):
        vk.vkDestroySemaphore(self.device, self.img_avail, None)
        vk.vkDestroySemaphore(self.device, self.render_done, None)
        vk.vkDestroyFence(self.device, self.in_flight, None)
        for fb in self.framebuffers:
            vk.vkDestroyFramebuffer(self.device, fb, None)
        vk.vkDestroySampler(self.device, self.sampler, None)
        vk.vkDestroyImageView(self.device, self.vol_view, None)
        vk.vkDestroyImage(self.device, self.vol_img, None)
        vk.vkFreeMemory(self.device, self.vol_mem, None)
        vk.vkDestroyImageView(self.device, self.obs_view, None)
        vk.vkDestroyImage(self.device, self.obs_img, None)
        vk.vkFreeMemory(self.device, self.obs_mem, None)
        vk.vkDestroyImageView(self.device, self.sdf_view, None)
        vk.vkDestroyImage(self.device, self.sdf_img, None)
        vk.vkFreeMemory(self.device, self.sdf_mem, None)
        self.params_buf.destroy()
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, None)
        vk.vkDestroyDescriptorSetLayout(self.device, self.desc_set_layout, None)
        vk.vkDestroyPipeline(self.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        vk.vkDestroyRenderPass(self.device, self.render_pass, None)
        for sv in self.swap_views:
            vk.vkDestroyImageView(self.device, sv, None)
        self.khr_destroy_swapchain(self.device, self.swapchain, None)
        self.khr_destroy_surface(self.instance, self.surface, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)
        glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == '__main__':
    import time
    viz = VKVisualizer()
    viz.run()
