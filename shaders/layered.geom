#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

flat in int v_instance_id[];
flat out int v_layer;

void main() {
    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        gl_Layer = v_instance_id[0];
        v_layer = v_instance_id[0];
        EmitVertex();
    }
    EndPrimitive();
}
