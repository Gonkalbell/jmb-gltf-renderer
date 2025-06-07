use std::f32::consts::TAU;

use eframe::egui::{self, DragValue, Widget};
use glam::{EulerRot, Mat4, Vec2, Vec3};
use puffin::profile_function;

/// A user-controlled camera that orbits around the origin.
#[derive(Debug)]
pub struct ArcBallCamera {
    center_pos: Vec3,
    pitch_revs: f32,
    yaw_revs: f32,
    dist: f32,

    aspect_ratio: f32,
    fov_y_revs: f32,
}

impl Default for ArcBallCamera {
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

impl ArcBallCamera {
    /// Call once per frame to update the camera's parameters with the current input state.
    pub fn update(&mut self, input: &egui::InputState) {
        profile_function!();

        let screen_size: Vec2 = <[f32; 2]>::from(input.screen_rect().size()).into();
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
            let clip_to_world = (self.projection_matrix() * self.view_matrix()).inverse();
            // I'm multiplying the dist here to "undo" the division that normally gets applied to perspective projection
            // And I'm making it negative because I want it to feel like "dragging" the camera, so the scene should move
            // in the opposite direction of the pointer.
            let pointer_delta = (-self.dist * clipspace_pointer_delta).extend(0.).extend(0.);
            let worldspace_pointer_delta = clip_to_world * pointer_delta;
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

    /// Get the premultiplied view and projection matrices.
    pub fn view_matrix(&self) -> Mat4 {
        let orbit = Mat4::from_translation(self.dist * Vec3::Z);
        let rot = Mat4::from_euler(
            EulerRot::YXZ,
            TAU * self.yaw_revs,
            TAU * self.pitch_revs,
            0.,
        );
        let translation = Mat4::from_translation(self.center_pos);
        (translation * rot * orbit).inverse()
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(TAU * self.fov_y_revs, self.aspect_ratio, 0.1, 1000.)
    }
}
