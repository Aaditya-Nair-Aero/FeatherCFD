import vulkan as vk

class VKBuffer:
    def __init__(self, ctx, size, usage, memory_properties):
        self.ctx = ctx
        self.size = size
        self.buffer = None
        self.memory = None
        self.mapped = None
        self._create_buffer(size, usage, memory_properties)

    def _create_buffer(self, size, usage, memory_properties):
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.buffer = vk.vkCreateBuffer(self.ctx.device, buffer_info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self.ctx.device, self.buffer)
        mem_type = self.ctx.find_memory_type(reqs.memoryTypeBits, memory_properties)
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type,
        )
        self.memory = vk.vkAllocateMemory(self.ctx.device, alloc_info, None)
        vk.vkBindBufferMemory(self.ctx.device, self.buffer, self.memory, 0)
        if memory_properties & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            self.mapped = vk.vkMapMemory(self.ctx.device, self.memory, 0, size, 0)

    def upload(self, data):
        if isinstance(data, bytes):
            self.mapped[:len(data)] = data
        else:
            self.mapped[:len(data)] = bytes(data)

    def download(self):
        return bytes(self.mapped[:self.size])

    def destroy(self):
        if self.mapped:
            vk.vkUnmapMemory(self.ctx.device, self.memory)
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)


def create_staging_buffer(ctx, size):
    return VKBuffer(ctx, size,
        vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)


def create_uniform_buffer(ctx, size):
    return VKBuffer(ctx, size,
        vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
