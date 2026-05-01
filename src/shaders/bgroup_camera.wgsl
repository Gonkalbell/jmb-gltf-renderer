#define_import_path bgroup_camera

struct Camera {
    world_to_local: mat4x4<f32>,
    local_to_world: mat4x4<f32>,
    local_to_proj: mat4x4<f32>,
    proj_to_local: mat4x4<f32>,
    world_to_proj: mat4x4<f32>,
    proj_to_world: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> res_camera: Camera;
