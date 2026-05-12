#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_init;
uniform sampler3D u_k1;
uniform sampler3D u_k2;
uniform sampler3D u_k3;
uniform sampler3D u_k4;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    vec3 init = texelFetch(u_init, texel, 0).xyz;
    vec3 k1 = texelFetch(u_k1, texel, 0).xyz;
    vec3 k2 = texelFetch(u_k2, texel, 0).xyz;
    vec3 k3 = texelFetch(u_k3, texel, 0).xyz;
    vec3 k4 = texelFetch(u_k4, texel, 0).xyz;
    vec3 combined = init + (k1 + 2.0*k2 + 2.0*k3 + k4) * (1.0/6.0);
    fragColor = vec4(combined, 1.0);
}
