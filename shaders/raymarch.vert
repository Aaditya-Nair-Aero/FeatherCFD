#version 330 core

// Full-screen quad generated procedurally from vertex ID.
// Outputs a ray direction in world space for the fragment shader.

out vec2 v_uv;       // screen UV [0..1]
out vec3 v_ray_dir;  // ray direction in world space (unnormalized)

// Camera uniforms
uniform vec3 u_cam_pos;       // Camera position in world space
uniform mat4 u_inv_view_proj; // Inverse of (projection * view)

void main() {
    // Two triangles covering NDC [-1, 1]
    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
    );
    vec2 ndc = positions[gl_VertexID];
    v_uv = ndc * 0.5 + 0.5;
    gl_Position = vec4(ndc, 0.0, 1.0);

    // Unproject NDC point on the far plane to get world-space ray direction
    vec4 world = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
    world.xyz /= world.w;
    v_ray_dir = world.xyz - u_cam_pos;
}
