#import bgroup_camera::res_camera

struct Material {
    base_color_factor: vec4f,
    alpha_cutoff: f32,
};

@group(1) @binding(0) var<uniform> material_data: Material;
@group(1) @binding(2) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

struct Instance {
    local_to_world: mat4x4f,
    normal_local_to_world: mat4x4f,
}
@group(2) @binding(0) var<storage> res_instances: array<Instance>;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec4f,
    @location(3) texcoord_0: vec2f,
    @location(4) texcoord_1: vec2f,
    @location(5) color_0: vec4f,
    @location(6) color_1: vec4f,
    // @location(7) joints_0: vec4f,
    // @location(8) joints_1: vec4f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) view_position: vec4f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) texcoord_0: vec2f,
    @location(4) texcoord_1: vec2f,
    @location(5) color_0: vec4f,
    @location(6) color_1: vec4f,
};

@vertex
fn vs_scene(@builtin(instance_index) instance_index: u32, input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var instance = res_instances[instance_index];
    output.position = res_camera.world_to_proj * instance.local_to_world * vec4f(input.position, 1);
    output.view_position = res_camera.world_to_local * instance.local_to_world * vec4f(input.position, 1);
    output.normal = (res_camera.world_to_local * instance.normal_local_to_world * vec4f(input.normal, 0)).xyz;
    output.tangent = (res_camera.world_to_local * instance.normal_local_to_world * vec4f(input.tangent.xyz, 0)).xyz;

    output.texcoord_0 = input.texcoord_0;
    output.texcoord_1 = input.texcoord_1;

    output.color_0 = input.color_0;
    output.color_1 = input.color_1;

    return output;
}

// Some hardcoded lighting
const LIGHT_DIR = vec3f(0.25, 0.5, 1);
const LIGHT_COLOR = vec3f(1);
const AMBIENT_COLOR = vec3f(0.1);

@fragment
fn fs_scene(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4f {
    let base_color = input.color_0 * textureSample(base_color_texture, base_color_sampler, input.texcoord_0) * material_data.base_color_factor;

    if (base_color.a < material_data.alpha_cutoff) {
        discard;
    }

    // An extremely simple directional lighting model, just to give our model some shape.
    var N = input.normal;

    // We use (0, 0, 0) as a sentinel value for if we need to compute flat normals manually
    if all(N == vec3f(0)) {
        let view_position = input.view_position.xyz;
        let dx = dpdx(view_position);
        let dy = dpdy(view_position);
        N = cross(dy, dx);
    }

    N = select(-1., 1., front_facing) * normalize(N);
    let L = normalize(LIGHT_DIR);
    let NDotL = max(dot(N, L), 0.0);
    let surface_color = (base_color.rgb * AMBIENT_COLOR) + (base_color.rgb * NDotL);

    return vec4f(surface_color, base_color.a);
}
