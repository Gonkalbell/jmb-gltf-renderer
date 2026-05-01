#![warn(clippy::all)]

mod app;

use std::future::Future;

pub use app::RendererApp;

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn(task: impl Future<Output = ()> + 'static + Send) {
    tokio::spawn(task);
}

#[cfg(target_arch = "wasm32")]
pub fn spawn(task: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(task);
}

mod asset;
mod camera;
mod skybox;

#[allow(clippy::all)]
mod shaders;

use std::sync::{Arc, Mutex};

use eframe::egui::Widget;
use eframe::{
    egui::{self, RichText, ahash::HashMap},
    egui_wgpu, wgpu,
};
use puffin::profile_function;
use reqwest::Url;
use serde::Deserialize;
use tokio::sync::watch::{Receiver, Sender};

use crate::{asset::{Asset, LoadingProgress}, camera::ArcBallCamera, skybox::Skybox};

const ASSETS_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub mod bind_groups {
    use super::shaders::*;

    pub type Camera = bgroup_camera::WgpuBindGroup0;
    pub type CameraEntries<'a> = bgroup_camera::WgpuBindGroup0Entries<'a>;
    pub type CameraEntriesParams<'a> = bgroup_camera::WgpuBindGroup0EntriesParams<'a>;

    pub type Instance = scene::WgpuBindGroup1;
    pub type InstanceEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
    pub type InstanceEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

    pub type Skybox = skybox::WgpuBindGroup1;
    pub type SkyboxEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
    pub type SkyboxEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;
}

pub struct SceneRenderer {
    camera: ArcBallCamera,

    skybox: Skybox,

    asset_rx: Receiver<Option<Asset>>,
    asset_tx: Sender<Option<Asset>>,

    asset_list: Receiver<Vec<ModelLinkInfo>>,
    loading_progress: Arc<Mutex<LoadingProgress>>,
}

impl SceneRenderer {
    pub fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let camera = ArcBallCamera::new(device);
        let skybox = Skybox::new(device, queue, color_format);

        let (asset_tx, asset_rx) = tokio::sync::watch::channel(None);
        let asset_tx_clone = asset_tx.clone();
        let loading_progress = Arc::new(Mutex::new(LoadingProgress {
            loaded: 1,
            total: 1,
        }));
        {
            let device = device.clone();
            let queue = queue.clone();
            let loading_progress_clone = loading_progress.clone();
            crate::spawn(async move {
                let url = Url::parse(ASSETS_BASE_URL)
                    .unwrap()
                    .join("AntiqueCamera/glTF-Binary/AntiqueCamera.glb")
                    .unwrap();
                let loaded_scene =
                    asset::load_asset(url, &device, &queue, color_format, loading_progress_clone)
                        .await
                        .unwrap();
                let _ = asset_tx_clone.send(Some(loaded_scene));
            });
        }

        // Load the asset list

        let (asset_list_tx, asset_list_rx) = tokio::sync::watch::channel(Vec::new());
        crate::spawn(async move {
            let url = Url::parse(ASSETS_BASE_URL)
                .unwrap()
                .join("model-index.json")
                .unwrap();
            let req: Vec<ModelLinkInfo> = reqwest::get(url).await.unwrap().json().await.unwrap();
            let req = req
                .into_iter()
                .filter(|m| m.tags.iter().any(|t| *t == "core"))
                .collect();
            let _ = asset_list_tx.send(req);
        });

        Self {
            camera,

            skybox,

            asset_rx,
            asset_tx,

            asset_list: asset_list_rx,
            loading_progress,
        }
    }

    pub fn prepare(
        &self,
        _device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
    ) -> Option<wgpu::CommandBuffer> {
        profile_function!();

        self.camera.update_buffer(queue);

        None
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        profile_function!();

        self.camera.bgroup.set(rpass);

        if let Some(asset) = self.asset_rx.borrow().as_ref() {
            asset.render(rpass);
        }

        self.skybox.render(rpass);
    }

    pub fn run_ui(&mut self, ui: &mut egui::Ui, render_state: &egui_wgpu::RenderState) {
        profile_function!();

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
    }

    fn show_info_menu(&mut self, render_state: &egui_wgpu::RenderState, ui: &mut egui::Ui) {
        if let Some(asset) = self.asset_rx.borrow().as_ref() {
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

    fn show_scene_menu(&mut self, render_state: &egui_wgpu::RenderState, ui: &mut egui::Ui) {
        let model_list = self.asset_list.borrow();
        if !model_list.is_empty() {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for model in model_list.iter() {
                    ui.menu_button(&model.label, |ui| {
                        for (variant, file) in &model.variants {
                            if ui.button(variant).clicked() {
                                let asset_tx = self.asset_tx.clone();
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
                                    let scene = asset::load_asset(
                                        url,
                                        &device,
                                        &queue,
                                        target_format,
                                        loading_progress,
                                    )
                                    .await
                                    .unwrap();
                                    let _ = asset_tx.send(Some(scene));
                                });
                            }
                        }
                    });
                }
            });
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelLinkInfo {
    label: String,
    name: String,
    #[serde(rename = "screenshot")]
    _screenshot: String,
    tags: Vec<String>,
    variants: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedBufferSlice {
    buffer: wgpu::Buffer,
    offset: wgpu::BufferAddress,
    size: wgpu::BufferSize,
}

impl OwnedBufferSlice {
    fn from_slice(slice: &wgpu::BufferSlice) -> Self {
        Self {
            buffer: slice.buffer().clone(),
            offset: slice.offset(),
            size: slice.size(),
        }
    }

    fn as_slice<'a>(&'a self) -> wgpu::BufferSlice<'a> {
        self.buffer
            .slice(self.offset..self.offset + self.size.get())
    }
}
