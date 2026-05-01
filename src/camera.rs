use std::f32::consts::TAU;

use eframe::egui::{self, DragValue, Widget};
use glam::{EulerRot, Mat4, Vec2, Vec3};
use puffin::profile_function;
use wgpu::util::DeviceExt;

use crate::{bind_groups, shaders::bgroup_camera};

pub struct ArcBallCamera {
    pub params: ArcBallCameraParams,
    pub bgroup: bind_groups::Camera,
    buffer: wgpu::Buffer,
}

impl ArcBallCamera {
    pub fn new(device: &wgpu::Device) -> Self {
        let params = ArcBallCameraParams::default();
        let buffer_data = params.get_buffer_data();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&buffer_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgroup = bind_groups::Camera::from_bindings(
            device,
            bind_groups::CameraEntries::new(bind_groups::CameraEntriesParams {
                res_camera: buffer.as_entire_buffer_binding(),
            }),
        );
        Self {
            params,
            buffer,
            bgroup,
        }
    }

    pub fn update_buffer(&self, queue: &wgpu::Queue) {
        let camera = self.params.get_buffer_data();
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&camera));
    }
}

/// A user-controlled camera that orbits around the origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcBallCameraParams {
    center_pos: Vec3,
    pitch_revs: f32,
    yaw_revs: f32,
    dist: f32,

    aspect_ratio: f32,
    fov_y_revs: f32,
}

impl Default for ArcBallCameraParams {
    fn default() -> Self {
        Self {
            center_pos: Vec3::ZERO,
            pitch_revs: 0.,
            yaw_revs: 0.,
            dist: 2.,

            aspect_ratio: 16. / 9.,
            fov_y_revs: 1. / 8.,
        }
    }
}

impl ArcBallCameraParams {
    /// Call once per frame to update the camera's parameters with the current input state.
    pub fn update(&mut self, input: &egui::InputState) {
        profile_function!();

        let screen_size: Vec2 = <[f32; 2]>::from(input.content_rect().size()).into();
        self.aspect_ratio = screen_size.x / screen_size.y;

        // Note: I'm using `delta` rather than `motion`. Even though `motion` is unfiltered and usually better suited for
        // 3D camera movement, I want camera movement to directly correspond to the cursors movement across the screen.
        let pointer_delta: Vec2 = <[f32; 2]>::from(input.pointer.delta()).into();

        // Invert Y, since I want a right-hand coord system and the mouse is in a left-hand coord system
        let clipspace_pointer_delta = Vec2::new(1., -1.) * (pointer_delta / screen_size);

        if input.pointer.primary_down() {
            let pitch_delta = clipspace_pointer_delta.y * self.fov_y_revs;
            self.pitch_revs = (self.pitch_revs + pitch_delta).clamp(-0.25, 0.25);

            let fov_x_revs =
                2. * ((TAU * self.fov_y_revs / 2.).tan() * self.aspect_ratio).atan() / TAU;
            let yaw_delta = -clipspace_pointer_delta.x * fov_x_revs;
            self.yaw_revs = (self.yaw_revs + yaw_delta).rem_euclid(1.);
        }

        if input.pointer.secondary_down() {
            let proj_to_world =
                self.local_to_world_matrix() * self.local_to_proj_matrix().inverse();
            // I'm multiplying the dist here to "undo" the division that normally gets applied to perspective projection
            // And I'm making it negative because I want it to feel like "dragging" the camera, so the scene should move
            // in the opposite direction of the pointer.
            let pointer_delta = (-self.dist * clipspace_pointer_delta).extend(0.).extend(0.);
            let worldspace_pointer_delta = proj_to_world * pointer_delta;
            self.center_pos += worldspace_pointer_delta.truncate();
        }

        let (_, scroll_y) = input.smooth_scroll_delta.into();
        self.dist -= scroll_y * input.stable_dt;

        self.fov_y_revs = (self.fov_y_revs / input.zoom_delta()).clamp(0.0001, 1. / 2.);
    }

    /// Show a gui window for modifying the camera parameters.
    pub fn run_ui(&mut self, ui: &mut egui::Ui) {
        profile_function!();

        egui::Grid::new("Camera").num_columns(2).show(ui, |ui| {
            ui.label("pitch");
            DragValue::new(&mut self.pitch_revs)
                .suffix("τ")
                .speed(0.01)
                .ui(ui);
            ui.end_row();
            self.pitch_revs = self.pitch_revs.clamp(-0.25, 0.25);

            ui.label("yaw");
            DragValue::new(&mut self.yaw_revs)
                .suffix("τ")
                .speed(0.01)
                .ui(ui);
            ui.end_row();
            self.yaw_revs = self.yaw_revs.rem_euclid(1.);

            ui.label("distance");
            DragValue::new(&mut self.dist).suffix("m").speed(0.1).ui(ui);
            ui.end_row();

            ui.label("vertical FOV");
            DragValue::new(&mut self.fov_y_revs)
                .suffix("τ")
                .speed(0.01)
                .range(0. ..=0.5)
                .ui(ui);
            ui.end_row();
        });
    }

    pub fn local_to_world_matrix(&self) -> Mat4 {
        let orbit = Mat4::from_translation(self.dist * Vec3::Z);
        let rot = Mat4::from_euler(
            EulerRot::YXZ,
            TAU * self.yaw_revs,
            TAU * self.pitch_revs,
            0.,
        );
        let translation = Mat4::from_translation(self.center_pos);
        translation * rot * orbit
    }

    pub fn local_to_proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(TAU * self.fov_y_revs, self.aspect_ratio, 0.1, 1000.)
    }

    pub fn get_buffer_data(&self) -> bgroup_camera::Camera {
        let local_to_world = self.local_to_world_matrix();
        let world_to_local = local_to_world.inverse();
        let local_to_proj = self.local_to_proj_matrix();
        let proj_to_local = local_to_proj.inverse();
        let world_to_proj = local_to_proj * world_to_local;
        let proj_to_world = local_to_world * proj_to_local;
        bgroup_camera::Camera {
            world_to_local,
            local_to_world,
            local_to_proj,
            proj_to_local,
            world_to_proj,
            proj_to_world,
        }
    }
}
