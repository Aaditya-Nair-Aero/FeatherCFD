import vulkan as vk

class VKDescriptorManager:
    def __init__(self, device, max_sets=1024):
        self.device = device
        self.max_sets = max_sets
        self.pool = None
        self._cache = {}
        self._create_pool()

    def _create_pool(self):
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=self.max_sets * 8),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=self.max_sets * 4),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=self.max_sets * 4),
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=self.max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

    def create_storage_image_layout(self, binding_count=1):
        bindings = []
        for i in range(binding_count):
            bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ))
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        return vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

    def create_descriptor_set(self, layout, image_views, sampler=None):
        key = (layout, tuple(image_views), sampler)
        if key in self._cache:
            return self._cache[key]

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.pool,
            descriptorSetCount=1,
            pSetLayouts=[layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info, None)[0]

        writes = []
        for i, view in enumerate(image_views):
            image_info = vk.VkDescriptorImageInfo(
                sampler=vk.VK_NULL_HANDLE if sampler is None else sampler,
                imageView=view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[image_info],
            ))
        if writes:
            vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        self._cache[key] = descriptor_set
        return descriptor_set

    def clear_cache(self):
        self._cache.clear()

    def destroy(self):
        self._cache.clear()
        vk.vkDestroyDescriptorPool(self.device, self.pool, None)
