mod camera;
mod gltf_loader;

#[allow(clippy::all)]
mod shaders;

use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use eframe::egui::Widget;
use eframe::{
    egui::{self, RichText, ahash::HashMap},
    egui_wgpu, wgpu,
};
use puffin::profile_function;
use reqwest::Url;
use serde::Deserialize;
use tokio::sync::watch::{Receiver, Sender};
use wgpu::util::DeviceExt;

use camera::ArcBallCamera;

use shaders::*;

const ASSETS_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

type CameraBindGroup = bgroup_camera::WgpuBindGroup0;
type CameraBindGroupEntries<'a> = bgroup_camera::WgpuBindGroup0Entries<'a>;
type CameraBindGroupEntriesParams<'a> = bgroup_camera::WgpuBindGroup0EntriesParams<'a>;

type InstanceBindGroup = scene::WgpuBindGroup1;
type InstanceBindGroupEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
type InstanceBindGroupEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

type SkyboxBindGroup = skybox::WgpuBindGroup1;
type SkyboxBindGroupEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
type SkyboxBindGroupEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;

pub struct SceneRenderer {
    camera_buf: wgpu::Buffer,
    user_camera: ArcBallCamera,
    camera_bgroup: CameraBindGroup,

    skybox_bgroup: SkyboxBindGroup,
    skybox_pipeline: wgpu::RenderPipeline,

    asset_rx: Receiver<Option<Asset>>,
    asset_tx: Sender<Option<Asset>>,

    asset_list: Receiver<Vec<ModelLinkInfo>>,
    loading_progress: Arc<Mutex<LoadingProgress>>,
}

struct LoadingProgress {
    loaded: usize,
    total: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Asset {
    asset_info: String,
    batches: Vec<RenderBatch>,
    instance_bgroup: InstanceBindGroup,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderBatch {
    pipeline: wgpu::RenderPipeline,
    mesh_primitives: Vec<Primitive>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Primitive {
    attrib_buffers: Vec<OwnedBufferSlice>,
    draw_count: u32,
    index_data: Option<PrimitiveIndexData>,
    instances: Range<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrimitiveIndexData {
    format: wgpu::IndexFormat,
    buffer_slice: OwnedBufferSlice,
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

#[derive(Debug, Deserialize)]
struct ModelLinkInfo {
    label: String,
    name: String,
    #[serde(rename = "screenshot")]
    _screenshot: String,
    tags: Vec<String>,
    variants: HashMap<String, String>,
}

impl SceneRenderer {
    pub fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // Camera

        let user_camera = ArcBallCamera::default();
        let camera_buffer_data = user_camera.get_buffer_data();

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&camera_buffer_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bgroup = CameraBindGroup::from_bindings(
            device,
            CameraBindGroupEntries::new(CameraBindGroupEntriesParams {
                res_camera: camera_buf.as_entire_buffer_binding(),
            }),
        );

        // Skybox

        let ktx_reader = ktx2::Reader::new(include_bytes!("../assets/rgba8.ktx2"))
            .expect("Failed to find skybox texture");
        let mut image = Vec::with_capacity(ktx_reader.data().len());
        for level in ktx_reader.levels() {
            image.extend_from_slice(level.data);
        }
        let ktx_header = ktx_reader.header();
        let skybox_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("../assets/rgba8.ktx2"),
                size: wgpu::Extent3d {
                    width: ktx_header.pixel_width,
                    height: ktx_header.pixel_height,
                    depth_or_array_layers: ktx_header.face_count,
                },
                mip_level_count: ktx_header.level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &image,
        );
        let skybox_tview = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("../assets/rgba8.ktx2"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });

        let skybox_bgroup = SkyboxBindGroup::from_bindings(
            device,
            SkyboxBindGroupEntries::new(SkyboxBindGroupEntriesParams {
                res_texture: &skybox_tview,
                res_sampler: &device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("skybox sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                }),
            }),
        );

        let shader = skybox::create_shader_module_embed_source(device);
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox"),
            layout: Some(&skybox::create_pipeline_layout(device)),
            vertex: skybox::vertex_state(&shader, &skybox::vs_skybox_entry()),
            fragment: Some(skybox::fragment_state(
                &shader,
                &skybox::fs_skybox_entry([Some(color_format.into())]),
            )),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Load the GLTF scene

        let (asset_tx, asset_rx) = tokio::sync::watch::channel(None);
        let asset_tx_clone = asset_tx.clone();
        let device = device.clone();
        let queue = queue.clone();
        let loading_progress = Arc::new(Mutex::new(LoadingProgress {
            loaded: 1,
            total: 1,
        }));
        {
            let loading_progress_clone = loading_progress.clone();
            crate::spawn(async move {
                let url = Url::parse(ASSETS_BASE_URL)
                    .unwrap()
                    .join("AntiqueCamera/glTF-Binary/AntiqueCamera.glb")
                    .unwrap();
                let loaded_scene = gltf_loader::load_asset(
                    url,
                    &device,
                    &queue,
                    color_format,
                    loading_progress_clone,
                )
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
            user_camera,
            camera_buf,
            camera_bgroup,

            skybox_bgroup,
            skybox_pipeline,

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

        let camera = self.user_camera.get_buffer_data();
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&camera));

        None
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        profile_function!();

        self.camera_bgroup.set(rpass);

        if let Some(asset) = self.asset_rx.borrow().as_ref() {
            asset.instance_bgroup.set(rpass);

            for batch in asset.batches.iter() {
                rpass.set_pipeline(&batch.pipeline);

                for primitive in batch.mesh_primitives.iter() {
                    for (i, attrib) in primitive.attrib_buffers.iter().enumerate() {
                        rpass.set_vertex_buffer(i as _, attrib.as_slice());
                    }

                    if let Some(index_data) = &primitive.index_data {
                        rpass.set_index_buffer(
                            index_data.buffer_slice.as_slice(),
                            index_data.format,
                        );
                    }

                    if primitive.index_data.is_none() {
                        rpass.draw(0..primitive.draw_count, primitive.instances.clone());
                    } else {
                        rpass.draw_indexed(0..primitive.draw_count, 0, primitive.instances.clone());
                    }
                }
            }
        }

        self.skybox_bgroup.set(rpass);
        rpass.set_pipeline(&self.skybox_pipeline);
        rpass.draw(0..3, 0..1);
    }

    pub fn run_ui(&mut self, ui: &mut egui::Ui, render_state: &egui_wgpu::RenderState) {
        profile_function!();

        let ctx = ui.ctx();
        if !ctx.egui_wants_keyboard_input() && !ctx.egui_wants_pointer_input() {
            ctx.input(|input| {
                self.user_camera.update(input);
            });
        }

        egui::Panel::top("top_panel").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Asset", |ui| {
                    self.show_scene_menu(render_state, ui);
                });

                ui.menu_button("Camera", |ui| self.user_camera.run_ui(ui));

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
                ui.label(&asset.asset_info);
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
                                    let scene = gltf_loader::load_asset(
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
