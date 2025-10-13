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


//        width
//          |
//          v
//start-->0---2,3
//        |  / |
//        | /  |
// end-->1,4---5
@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let line = lines[iid];

    var pos : vec2<f32>;

    let dir = normalize(line.end - line.start);
    let perp = line.width * vec2f(-dir.y, dir.x);

    switch (vid) {
        case 0u: {
            pos = line.start - perp;
        }
        case 1u, 4u: {
            pos = line.end - perp;
        }
        case 2u, 3u: {
            pos = line.start + perp;
        }
        case 5u: {
            pos = line.end + perp;
        }
        default: {
            pos = vec2<f32>(0.0, 0.0);
        }
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
