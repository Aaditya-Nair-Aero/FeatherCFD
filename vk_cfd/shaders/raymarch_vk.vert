#version 450 core

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_ray_dir;

layout(push_constant) uniform PushConstants {
    vec3 cam_pos;
    float pad0;
    mat4 inv_view_proj;
} pc;

void main() {
    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
    );
    vec2 ndc = positions[gl_VertexIndex];
    v_uv = ndc * 0.5 + 0.5;
    gl_Position = vec4(ndc, 0.0, 1.0);

    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
    world.xyz /= world.w;
    v_ray_dir = world.xyz - pc.cam_pos;
}
