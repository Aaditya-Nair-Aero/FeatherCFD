#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_coarse;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    ivec3 coarse_sz = textureSize(u_coarse, 0);

    vec3 pos = vec3(texel) / 2.0;
    ivec3 c0 = ivec3(floor(pos));
    c0 = clamp(c0, ivec3(0, 0, 0), coarse_sz - 2);
    vec3 t = clamp(pos - vec3(c0), vec3(0.0), vec3(1.0));

    float v000 = texelFetch(u_coarse, ivec3(c0.x,     c0.y,     c0.z),     0).x;
    float v100 = texelFetch(u_coarse, ivec3(c0.x + 1, c0.y,     c0.z),     0).x;
    float v010 = texelFetch(u_coarse, ivec3(c0.x,     c0.y + 1, c0.z),     0).x;
    float v110 = texelFetch(u_coarse, ivec3(c0.x + 1, c0.y + 1, c0.z),     0).x;
    float v001 = texelFetch(u_coarse, ivec3(c0.x,     c0.y,     c0.z + 1), 0).x;
    float v101 = texelFetch(u_coarse, ivec3(c0.x + 1, c0.y,     c0.z + 1), 0).x;
    float v011 = texelFetch(u_coarse, ivec3(c0.x,     c0.y + 1, c0.z + 1), 0).x;
    float v111 = texelFetch(u_coarse, ivec3(c0.x + 1, c0.y + 1, c0.z + 1), 0).x;

    float v00 = mix(v000, v100, t.x);
    float v01 = mix(v001, v101, t.x);
    float v10 = mix(v010, v110, t.x);
    float v11 = mix(v011, v111, t.x);
    float v0  = mix(v00,  v10,  t.y);
    float v1  = mix(v01,  v11,  t.y);
    float v   = mix(v0,   v1,   t.z);

    fragColor = v;
}
