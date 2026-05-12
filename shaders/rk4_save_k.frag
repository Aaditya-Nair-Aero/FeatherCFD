#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_result;
uniform sampler3D u_input;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    vec3 result = texelFetch(u_result, texel, 0).xyz;
    vec3 inp = texelFetch(u_input, texel, 0).xyz;
    fragColor = vec4(result - inp, 1.0);
}
