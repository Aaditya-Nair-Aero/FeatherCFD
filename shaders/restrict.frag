#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_fine;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    ivec3 fine_sz = textureSize(u_fine, 0);

    ivec3 base = 2 * texel;

    float sum = 0.0;
    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for (int dk = 0; dk < 2; dk++) {
                ivec3 fpos = min(base + ivec3(di, dj, dk), fine_sz - 1);
                sum += texelFetch(u_fine, fpos, 0).x;
            }
        }
    }
    fragColor = sum / 8.0;
}
