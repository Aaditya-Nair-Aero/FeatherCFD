import subprocess
import os
import tempfile
import vulkan as vk

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shaders')
SPIRV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spirv')

def get_glslang_path():
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin', 'glslangValidator')
    if os.path.exists(local):
        return local
    return 'glslangValidator'

def compile_glsl_to_spirv(glsl_source, shader_stage='compute'):
    stage_ext = {
        'compute': '.comp',
        'vertex': '.vert',
        'fragment': '.frag',
        'geometry': '.geom',
    }
    ext = stage_ext[shader_stage]

    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
        f.write(glsl_source)
        glsl_path = f.name

    spv_path = glsl_path + '.spv'
    glslang = get_glslang_path()

    try:
        result = subprocess.run(
            [glslang, '-V', '-o', spv_path, glsl_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"glslangValidator failed:\n{result.stderr}\n{result.stdout}")
        with open(spv_path, 'rb') as f:
            spirv = f.read()
    finally:
        os.unlink(glsl_path)
        if os.path.exists(spv_path):
            os.unlink(spv_path)

    return spirv


def compile_file_to_spirv(filepath, shader_stage='compute'):
    with open(filepath, 'r') as f:
        source = f.read()
    return compile_glsl_to_spirv(source, shader_stage)


def create_shader_module(device, spirv_bytes):
    words = len(spirv_bytes) // 4
    create_info = vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(spirv_bytes),
        pCode=spirv_bytes,
    )
    return vk.vkCreateShaderModule(device, create_info, None)
