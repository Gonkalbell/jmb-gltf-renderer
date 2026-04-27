#import bgroup_camera::res_camera

struct Instance {
    local_to_world: mat4x4f,
    normal_local_to_world: mat4x4f,
}
@group(1) @binding(0) var<storage> res_instances : array<Instance>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) texcoord_0: vec2<f32>,
    @location(4) texcoord_1: vec2<f32>,
    @location(5) color_0: vec4<f32>,
    @location(6) color_1: vec4<f32>,
    // @location(5) joints_0: vec4<f32>,
    // @location(7) joints_1: vec4<f32>,
    // @location(5) weights_0: vec4<f32>,
    // @location(8) weights_1: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) view_position: vec4f,
    @location(1) normal: vec3f,
};

@vertex
fn vs_scene(@builtin(instance_index) instance_index: u32, input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var instance = res_instances[instance_index];
    output.position = res_camera.world_to_proj * instance.local_to_world * vec4f(input.position, 1);
    output.normal = (res_camera.world_to_local * instance.normal_local_to_world * vec4f(input.normal, 0)).xyz;

    output.view_position = res_camera.world_to_local * instance.local_to_world * vec4f(input.position, 1);

    return output;
}

// Some hardcoded lighting
const LIGHT_DIR = vec3f(0.25, 0.5, 1);
const AMBIENT_COLOR = vec3f(0.1);

@fragment
fn fs_scene(input: VertexOutput) -> @location(0) vec4f {
    // An extremely simple directional lighting model, just to give our model some shape.
    var N = input.normal;

    // We use (0, 0, 0) as a sentinel value for if we need to compute flat normals manually
    if all(N == vec3f(0)) {
        let view_position = input.view_position.xyz;
        let dx = dpdx(view_position);
        let dy = dpdy(view_position);
        N = cross(dy, dx);
    }

    N = normalize(N);
    let L = normalize(LIGHT_DIR);
    let NDotL = max(dot(N, L), 0.0);
    let surface_color = AMBIENT_COLOR + NDotL;

    return vec4f((1.f + N) / 2.f, 1);
}