//! Since I mostly want to do my own rendering, very little actually happens in the top level `App` struct. Instead,
//! most of the rendering logic actually happens in `renderer.rs`

use std::sync::{Arc, Mutex, RwLock};

use eframe::{
    egui::{self, RichText, Widget, ahash::HashMap},
    egui_wgpu::{self, CallbackTrait, RenderState},
};
use puffin::profile_function;
use reqwest::Url;
use serde::Deserialize;

use crate::{
    ASSETS_BASE_URL,
    asset::{self, Asset, LoadingProgress},
    camera::ArcBallCamera,
    skybox::Skybox,
};

#[derive(Debug, Deserialize)]
pub struct ModelLinkInfo {
    label: String,
    name: String,
    #[serde(rename = "screenshot")]
    _screenshot: String,
    tags: Vec<String>,
    variants: HashMap<String, String>,
}

pub struct RendererApp {
    camera: ArcBallCamera,

    skybox: Skybox,

    asset: Arc<RwLock<Option<Asset>>>,

    asset_list: Arc<RwLock<Vec<ModelLinkInfo>>>,
    loading_progress: Arc<Mutex<LoadingProgress>>,
}

impl RendererApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let asset_list = Arc::new(RwLock::new(Vec::new()));
        {
            let asset_list = asset_list.clone();
            crate::spawn(async move {
                let url = Url::parse(ASSETS_BASE_URL)
                    .unwrap()
                    .join("model-index.json")
                    .unwrap();
                let assets: Vec<ModelLinkInfo> =
                    reqwest::get(url).await.unwrap().json().await.unwrap();
                let assets = assets
                    .into_iter()
                    .filter(|m| m.tags.iter().any(|t| *t == "core"))
                    .collect();
                if let Ok(mut writer) = asset_list.write() {
                    *writer = assets;
                };
            });
        }

        let loading_progress = Arc::new(Mutex::new(LoadingProgress {
            loaded: 1,
            total: 1,
        }));

        // Initialize the renderer
        let wgpu_render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("WGPU is not properly initialized");

        let RenderState {
            device,
            queue,
            target_format,
            ..
        } = wgpu_render_state.clone();

        let camera = ArcBallCamera::new(&device);
        let skybox = Skybox::new(&device, &queue, target_format);

        let asset = Arc::new(RwLock::new(None));
        {
            let loading_progress = loading_progress.clone();
            let asset = asset.clone();
            crate::spawn(async move {
                let url = Url::parse(ASSETS_BASE_URL)
                    .unwrap()
                    .join("AntiqueCamera/glTF-Binary/AntiqueCamera.glb")
                    .unwrap();
                let loaded_asset =
                    asset::load_asset(url, &device, &queue, target_format, loading_progress)
                        .await
                        .unwrap();
                if let Ok(mut writer) = asset.write() {
                    *writer = Some(loaded_asset);
                };
            });
        }

        Self {
            camera,
            skybox,
            asset,
            asset_list,
            loading_progress,
        }
    }

    fn show_info_menu(&self, render_state: &eframe::egui_wgpu::RenderState, ui: &mut egui::Ui) {
        if let Ok(Some(asset)) = self.asset.read().as_deref() {
            ui.menu_button("Asset", |ui| {
                ui.label(asset.info());
            });
        }
        ui.menu_button("Adapter", |ui| {
            let info = render_state.adapter.get_info();
            ui.label(format!("name: {}", info.name));
            ui.label(format!("backend: {}", info.backend));
            ui.label(format!("driver: {}", info.driver));
            ui.label(format!("driver info: {}", info.driver_info));
            ui.label(format!("type: {:?}", info.device_type));
        });
        ui.menu_button("Counters", |ui| {
            let counters = render_state.device.get_internal_counters().hal;
            ui.label(format!("buffers: {}", counters.buffers.read()));
            ui.label(format!("textures: {}", counters.textures.read()));
            ui.label(format!("texture_views: {}", counters.texture_views.read()));
            ui.label(format!("bind_groups: {}", counters.bind_groups.read()));
            ui.label(format!(
                "bind_group_layouts: {}",
                counters.bind_group_layouts.read()
            ));
            ui.label(format!(
                "render_pipelines: {}",
                counters.render_pipelines.read()
            ));
            ui.label(format!(
                "compute_pipelines: {}",
                counters.compute_pipelines.read()
            ));
            ui.label(format!(
                "pipeline_layouts: {}",
                counters.pipeline_layouts.read()
            ));
            ui.label(format!("samplers: {}", counters.samplers.read()));
            ui.label(format!(
                "command_encoders: {}",
                counters.command_encoders.read()
            ));
            ui.label(format!(
                "shader_modules: {}",
                counters.shader_modules.read()
            ));
            ui.label(format!("query_sets: {}", counters.query_sets.read()));
            ui.label(format!("fences: {}", counters.fences.read()));
            ui.label(format!("buffer_memory: {}", counters.buffer_memory.read()));
            ui.label(format!(
                "texture_memory: {}",
                counters.texture_memory.read()
            ));
            ui.label(format!(
                "acceleration_structure_memory: {}",
                counters.acceleration_structure_memory.read()
            ));
            ui.label(format!(
                "memory_allocations: {}",
                counters.memory_allocations.read()
            ));
        });

        if let Some(report) = render_state.device.generate_allocator_report() {
            ui.menu_button("Allocation Report", |ui| {
                for (i, block) in report.blocks.iter().enumerate() {
                    ui.menu_button(format!("block {}: {}", i, block.size), |ui| {
                        let mut sorted_allocations =
                            report.allocations[block.allocations.clone()].to_owned();
                        sorted_allocations.sort_by_key(|a| a.offset);
                        for allocation in sorted_allocations.iter() {
                            ui.label(
                                RichText::new(format!(
                                    "{:08X}-{:08X} {}",
                                    allocation.offset,
                                    allocation.offset + allocation.size,
                                    allocation.name
                                ))
                                .monospace(),
                            );
                        }
                    });
                }
                ui.label(format!(
                    "total_allocated_bytes: {}",
                    report.total_allocated_bytes
                ));
                ui.label(format!(
                    "total_allocated_bytes: {}",
                    report.total_reserved_bytes
                ));
            });
        }
    }

    fn show_scene_menu(&self, render_state: &eframe::egui_wgpu::RenderState, ui: &mut egui::Ui) {
        if let Ok(asset_list) = self.asset_list.read().as_deref()
            && !asset_list.is_empty()
        {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for model in asset_list.iter() {
                    ui.menu_button(&model.label, |ui| {
                        for (variant, file) in &model.variants {
                            if ui.button(variant).clicked() {
                                let asset = self.asset.clone();
                                let egui_wgpu::RenderState {
                                    device,
                                    queue,
                                    target_format,
                                    ..
                                } = render_state.clone();
                                let url = Url::parse(ASSETS_BASE_URL)
                                    .unwrap()
                                    .join(&format!("{}/{}/{}", &model.name, variant, file))
                                    .unwrap();
                                let loading_progress = self.loading_progress.clone();
                                crate::spawn(async move {
                                    let loaded_asset = asset::load_asset(
                                        url,
                                        &device,
                                        &queue,
                                        target_format,
                                        loading_progress,
                                    )
                                    .await
                                    .unwrap();
                                    if let Ok(mut writer) = asset.write() {
                                        *writer = Some(loaded_asset);
                                    };
                                });
                            }
                        }
                    });
                }
            });
        }
    }
}

impl eframe::App for RendererApp {
    /// Called when the UI is being composed. This is where all rendering happens.
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        profile_function!();
        let render_state = frame
            .wgpu_render_state()
            .expect("WGPU is not properly initialized");

        let ctx = ui.ctx();
        if !ctx.egui_wants_keyboard_input() && !ctx.egui_wants_pointer_input() {
            ctx.input(|input| {
                self.camera.params.update(input);
            });
        }

        egui::Panel::top("top_panel").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Asset", |ui| {
                    self.show_scene_menu(render_state, ui);
                });

                ui.menu_button("Camera", |ui| self.camera.params.run_ui(ui));

                ui.menu_button("Info", |ui| {
                    self.show_info_menu(render_state, ui);
                });
                if let Ok(loading_progress) = self.loading_progress.try_lock()
                    && loading_progress.loaded != loading_progress.total
                {
                    let progress = loading_progress.loaded as f32 / loading_progress.total as f32;
                    egui::ProgressBar::new(progress)
                        .text(format!(
                            "loading {} / {}",
                            loading_progress.loaded, loading_progress.total
                        ))
                        .animate(true)
                        .ui(ui);
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    ui.viewport_rect(),
                    RenderCallback {
                        camera: self.camera.clone(),
                        skybox: self.skybox.clone(),
                        asset: self.asset.clone(),
                    },
                ));
        });
    }
}

// TODO: While `eframe` does handle a lot of the boilerplate for me, it wasn't really meant for a situation where I am
// mostly doing my own custom rendering. The main challenge is that the only way to do custom rendering is through a
// struct that implements `CallbackTrait`, which I have several nitpicks with:
//   - I don't have direct access to the `wgpu::Surface` or `wgpu::SurfaceTexture`. The `render` function uses the same
//     `wgpu::RenderPass` that the rest of egui uses to render to the surface, but I can't make multiple
//     `wgpu::RenderPass`s that all target the `wgpu::SurfaceTexture`
//   - `CustomCallback` must be recreated every frame. In fact `new_paint_callback` allocates a new Arc every frame.
// If any of these become a deal breaker, I may consider just using `winit` and `egui` directly. .
struct RenderCallback {
    camera: ArcBallCamera,

    skybox: Skybox,

    asset: Arc<RwLock<Option<Asset>>>,
}

impl CallbackTrait for RenderCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        profile_function!();

        self.camera.bgroup.set(render_pass);

        if let Ok(Some(asset)) = self.asset.read().as_deref() {
            asset.render(render_pass);
        }

        self.skybox.render(render_pass);
    }

    fn prepare(
        &self,
        _device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
        _callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        profile_function!();
        self.camera.update_buffer(queue);
        Vec::new()
    }
}
