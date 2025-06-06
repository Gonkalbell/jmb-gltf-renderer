mod camera;
mod gltf_loader;

#[allow(clippy::all)]
mod shaders;

use std::sync::{Arc, Mutex};

use eframe::wgpu;
use egui::{ahash::HashMap};
use puffin::profile_function;
use reqwest::Url;
use serde::Deserialize;
use wgpu::util::DeviceExt;

use camera::ArcBallCamera;

use shaders::*;

const ASSETS_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

type CameraBindGroup = bgroup_camera::WgpuBindGroup0;
type CameraBindGroupEntries<'a> = bgroup_camera::WgpuBindGroup0Entries<'a>;
type CameraBindGroupEntriesParams<'a> = bgroup_camera::WgpuBindGroup0EntriesParams<'a>;

type NodeBindGroup = scene::WgpuBindGroup1;
type NodeBindGroupEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
type NodeBindGroupEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

type SkyboxBindGroup = skybox::WgpuBindGroup1;
type SkyboxBindGroupEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
type SkyboxBindGroupEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;

pub struct SceneRenderer {
    camera_buf: wgpu::Buffer,
    user_camera: ArcBallCamera,
    camera_bgroup: CameraBindGroup,

    skybox_bgroup: SkyboxBindGroup,
    skybox_pipeline: wgpu::RenderPipeline,

    scene: Arc<Mutex<Option<Scene>>>,

    asset_list: Arc<Mutex<Vec<ModelLinkInfo>>>,
}

struct Scene {
    nodes: Vec<Node>,
    meshes: Vec<Mesh>,
}

struct Node {
    mesh_index: usize,
    bgroup: NodeBindGroup,
}

#[derive(Debug)]
struct Mesh {
    primitives: Vec<Primitive>,
}

#[derive(Debug)]
struct Primitive {
    pipeline: wgpu::RenderPipeline,
    attrib_buffers: Vec<AttribBuffer>,
    draw_count: u32,
    index_data: Option<PrimitiveIndexData>,
}

#[derive(Debug)]
struct AttribBuffer {
    buffer: Arc<wgpu::Buffer>,
    offset: wgpu::BufferAddress,
}

#[derive(Debug)]
struct PrimitiveIndexData {
    buffer: Arc<wgpu::Buffer>,
    format: wgpu::IndexFormat,
    offset: u64,
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

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&bgroup_camera::Camera {
                view: Default::default(),
                view_inv: Default::default(),
                proj: Default::default(),
                proj_inv: Default::default(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bgroup = CameraBindGroup::from_bindings(
            &device,
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
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
            }),
        );

        let shader = skybox::create_shader_module_embed_source(&device);
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox"),
            layout: Some(&skybox::create_pipeline_layout(&device)),
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load the GLTF scene

        let scene = Arc::new(Mutex::new(None));
        let scene_clone = scene.clone();
        let device = device.clone();
        crate::spawn(async move {
            let url = Url::parse(ASSETS_BASE_URL)
                .unwrap()
                .join("AntiqueCamera/glTF-Binary/AntiqueCamera.glb")
                .unwrap();
            let loaded_scene = gltf_loader::load_asset(url, &device, color_format)
                .await
                .unwrap();
            scene_clone.lock().unwrap().replace(loaded_scene);
        });

        // Load the asset list

        let asset_list = Arc::new(Mutex::new(Vec::new()));
        let asset_list_clone = asset_list.clone();
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
            *asset_list_clone.lock().unwrap() = req;
        });

        Self {
            user_camera,
            camera_buf,
            camera_bgroup,

            skybox_bgroup,
            skybox_pipeline,

            scene,

            asset_list,
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

        let view = self.user_camera.view_matrix();
        let proj = self.user_camera.projection_matrix();
        let camera = bgroup_camera::Camera {
            view,
            view_inv: view.inverse(),
            proj,
            proj_inv: proj.inverse(),
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&camera));

        None
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        profile_function!();

        self.camera_bgroup.set(rpass);

        let scene_lock = self.scene.try_lock();
        if let Ok(Some(scene)) = scene_lock.as_ref().map(|x| x.as_ref()) {
            for node in &scene.nodes {
                node.bgroup.set(rpass);
                let mesh = scene
                    .meshes
                    .get(node.mesh_index)
                    .expect("Node didn't have a mesh");
                for primitive in &mesh.primitives {
                    rpass.set_pipeline(&primitive.pipeline);
                    for (i, attrib) in primitive.attrib_buffers.iter().enumerate() {
                        rpass.set_vertex_buffer(i as _, attrib.buffer.slice(attrib.offset..));
                    }
                    if let Some(index_data) = &primitive.index_data {
                        rpass.set_index_buffer(
                            index_data.buffer.slice(index_data.offset..),
                            index_data.format,
                        );
                        rpass.draw_indexed(0..primitive.draw_count, 0, 0..1);
                    } else {
                        rpass.draw(0..primitive.draw_count, 0..1);
                    }
                }
            }
        }

        self.skybox_bgroup.set(rpass);
        rpass.set_pipeline(&self.skybox_pipeline);
        rpass.draw(0..3, 0..1);
    }

    pub fn run_ui(
        &mut self,
        ctx: &egui::Context,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
    ) {
        profile_function!();

        if !ctx.wants_keyboard_input() && !ctx.wants_pointer_input() {
            ctx.input(|input| {
                self.user_camera.update(input);
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Scene", |ui| {
                    self.show_scene_menu(device, color_format, ui);
                });

                ui.menu_button("Camera", |ui| self.user_camera.run_ui(ui));
            });
        });
    }

    fn show_scene_menu(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        ui: &mut egui::Ui,
    ) {
        if let Ok(model_list) = self.asset_list.try_lock() {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for model in model_list.iter() {
                    ui.collapsing(&model.label, |ui| {
                        for (variant, file) in &model.variants {
                            if ui.button(variant).clicked() {
                                let scene_clone = self.scene.clone();
                                let device = device.clone();
                                let url = Url::parse(ASSETS_BASE_URL)
                                    .unwrap()
                                    .join(&format!("{}/{}/{}", &model.name, variant, file))
                                    .unwrap();
                                crate::spawn(async move {
                                    let scene = gltf_loader::load_asset(url, &device, color_format)
                                        .await
                                        .unwrap();
                                    scene_clone.lock().unwrap().replace(scene);
                                });
                            }
                        }
                    });
                }
            });
        } else {
            ui.label("loading");
        }
    }
}
