import vulkan as vk
import numpy as np

VK_FORMAT_MAP = {
    'rgba16f': vk.VK_FORMAT_R16G16B16A16_SFLOAT,
    'r32f': vk.VK_FORMAT_R32_SFLOAT,
    'r16f': vk.VK_FORMAT_R16_SFLOAT,
    'r8ui': vk.VK_FORMAT_R8_UINT,
}

NP_DTYPE_MAP = {
    vk.VK_FORMAT_R16G16B16A16_SFLOAT: np.float16,
    vk.VK_FORMAT_R32_SFLOAT: np.float32,
    vk.VK_FORMAT_R16_SFLOAT: np.float16,
    vk.VK_FORMAT_R8_UINT: np.uint8,
}

COMPONENT_MAP = {
    vk.VK_FORMAT_R16G16B16A16_SFLOAT: 4,
    vk.VK_FORMAT_R32_SFLOAT: 1,
    vk.VK_FORMAT_R16_SFLOAT: 1,
    vk.VK_FORMAT_R8_UINT: 1,
}

class VKImage3D:
    def __init__(self, ctx, fmt_name, size=192):
        self.ctx = ctx
        self.size = size
        self.format_name = fmt_name
        self.vk_format = VK_FORMAT_MAP[fmt_name]
        self.components = COMPONENT_MAP[self.vk_format]
        self.image = None
        self.view = None
        self.memory = None
        self._create_image()
        self._create_view()

    def _create_image(self):
        extent = vk.VkExtent3D(width=self.size, height=self.size, depth=self.size)
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_3D,
            format=self.vk_format,
            extent=extent,
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.image = vk.vkCreateImage(self.ctx.device, image_info, None)
        reqs = vk.vkGetImageMemoryRequirements(self.ctx.device, self.image)
        mem_type = self.ctx.find_memory_type(reqs.memoryTypeBits,
                                              vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type,
        )
        self.memory = vk.vkAllocateMemory(self.ctx.device, alloc_info, None)
        vk.vkBindImageMemory(self.ctx.device, self.image, self.memory, 0)
        self._transition_layout(vk.VK_IMAGE_LAYOUT_GENERAL)

    def _create_view(self):
        view_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_3D,
            format=self.vk_format,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        self.view = vk.vkCreateImageView(self.ctx.device, view_info, None)

    def _transition_layout(self, new_layout):
        cmd = self.ctx.create_command_buffer()
        self.ctx.begin_command_buffer(cmd)

        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )

        vk.vkCmdPipelineBarrier(cmd,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier])

        self.ctx.end_and_submit(cmd)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def upload(self, data):
        np_dtype = NP_DTYPE_MAP[self.vk_format]
        expected = (self.size,) * 3 if self.components == 1 else (self.size,) * 3 + (self.components,)
        # Vulkan expects [z][y][x][c] layout (x-fastest), numpy uses [x][y][z][c] (z-fastest)
        # Transpose axes 0 and 2 to match Vulkan's expected buffer layout
        if data.ndim == 4:
            arr = np.ascontiguousarray(np.transpose(data, (2, 1, 0, 3)), dtype=np_dtype)
        else:
            arr = np.ascontiguousarray(np.transpose(data, (2, 1, 0)), dtype=np_dtype)
        byte_size = arr.nbytes

        staging = self.ctx.create_buffer(byte_size,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        staging.upload(arr.tobytes())

        cmd = self.ctx.create_command_buffer()
        self.ctx.begin_command_buffer(cmd)

        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [barrier])

        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.size, height=self.size, depth=self.size),
        )
        vk.vkCmdCopyBufferToImage(cmd, staging.buffer, self.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [region])

        barrier2 = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier2])

        self.ctx.end_and_submit(cmd)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        staging.destroy()

    def download(self):
        np_dtype = NP_DTYPE_MAP[self.vk_format]
        shape = (self.size,) * 3 if self.components == 1 else (self.size,) * 3 + (self.components,)
        byte_size = int(np.prod(shape) * np.dtype(np_dtype).itemsize)

        staging = self.ctx.create_buffer(byte_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        cmd = self.ctx.create_command_buffer()
        self.ctx.begin_command_buffer(cmd)

        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [barrier])

        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.size, height=self.size, depth=self.size),
        )
        vk.vkCmdCopyImageToBuffer(cmd, self.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1, [region])

        barrier2 = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier2])

        self.ctx.end_and_submit(cmd)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

        data = staging.download()
        staging.destroy()
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        # Vulkan gives [z][y][x][c], transpose to [x][y][z][c] for numpy
        if arr.ndim == 4:
            arr = np.transpose(arr, (2, 1, 0, 3))
        else:
            arr = np.transpose(arr, (2, 1, 0))
        return arr

    def destroy(self):
        vk.vkDestroyImageView(self.ctx.device, self.view, None)
        vk.vkDestroyImage(self.ctx.device, self.image, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)
