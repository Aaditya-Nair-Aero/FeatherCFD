import vulkan as vk

class VKComputePipeline:
    def __init__(self, device, shader_module, descriptor_set_layout,
                 push_constant_size=0):
        self.device = device
        self.pipeline = None
        self.layout = None

        push_ranges = []
        if push_constant_size > 0:
            push_ranges.append(vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size,
            ))

        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1 if descriptor_set_layout else 0,
            pSetLayouts=[descriptor_set_layout] if descriptor_set_layout else None,
            pushConstantRangeCount=len(push_ranges),
            pPushConstantRanges=push_ranges if push_ranges else None,
        )
        self.layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName='main',
        )

        create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.layout,
        )
        self.pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE,
                                                      1, [create_info], None)[0]

    def bind(self, cmd_buffer):
        vk.vkCmdBindPipeline(cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)

    def destroy(self):
        vk.vkDestroyPipeline(self.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.device, self.layout, None)
