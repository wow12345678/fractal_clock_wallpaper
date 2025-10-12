// Vertex shader

struct LineData {
    start: vec2<f32>,
    end: vec2<f32>,
    color: vec3<f32>,
    width: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@group(0) @binding(0) var<storage,read> lines: array<LineData>;


// vid:
// 0..2 is first triangle
// 3..5 is second triangle
//    0---1,5
//    |  / |
//    | /  |
//   2,3---4
@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    
    let line = lines[iid];

    var pos : vec2f;

    switch (vid){
        case 0u: {
            pos = line.start;
        }
        case 1u,5u: {}
        case 2u,3u: {
            pos = line.end;
        }
        case 4u: {}
        default: {}
    }

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.color = line.color;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
