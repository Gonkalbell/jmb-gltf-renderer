#import bgroup_camera::res_camera

@group(1) @binding(0) var res_texture: texture_cube<f32>;
@group(1) @binding(1) var res_sampler: sampler;

struct SkyboxInterp {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
};

@vertex
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxInterp {
    // hacky way to draw a large triangle
    let tmp1 = i32(vertex_index) / 2;
    let tmp2 = i32(vertex_index) & 1;
    let pos = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        1.0,
        1.0
    );

    var result: SkyboxInterp;
    result.position = pos;
    let dir = vec4<f32>((res_camera.proj_to_local * pos).xyz, 0.0);
    result.tex_coord = (res_camera.local_to_world * dir).xyz;
    return result;
}

@fragment
fn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4<f32> {
    return textureSample(res_texture, res_sampler, vertex.tex_coord);
}
