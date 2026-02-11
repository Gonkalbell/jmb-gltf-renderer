mod camera;
// mod gltf_loader;

#[allow(clippy::all)]
mod shaders;

use std::borrow::Cow;

use eframe::egui::{self, ahash::HashMap};
use eframe::{egui_wgpu, wgpu};
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

type SkyboxBindGroup = skybox::WgpuBindGroup1;
type SkyboxBindGroupEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
type SkyboxBindGroupEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;

pub struct SceneRenderer {
    camera_buf: wgpu::Buffer,
    user_camera: ArcBallCamera,
    camera_bgroup: CameraBindGroup,

    skybox_bgroup: SkyboxBindGroup,
    skybox_pipeline: wgpu::RenderPipeline,

    raytrace_pipeline: wgpu::RenderPipeline,
    raytrace_default_bgroup: wgpu::BindGroup,

    asset_rx: Receiver<Option<Asset>>,
    asset_tx: Sender<Option<Asset>>,

    asset_list: Receiver<Vec<ModelLinkInfo>>,
}

struct Asset {
    scene: wgpu::Tlas,
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
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
            }),
        );

        let shader = skybox::create_shader_module_embed_source(device);
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
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
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load the GLTF scene

        let (asset_tx, asset_rx) = tokio::sync::watch::channel(None);
        let asset_tx_clone = asset_tx.clone();
        let device = device.clone();
        // crate::spawn(async move {
        //     let url = Url::parse(ASSETS_BASE_URL)
        //         .unwrap()
        //         .join("AntiqueCamera/glTF-Binary/AntiqueCamera.glb")
        //         .unwrap();
        //     let loaded_scene = gltf_loader::load_asset(url, &device, color_format)
        //         .await
        //         .unwrap();
        //     let _ = asset_tx_clone.send(Some(loaded_scene));
        // });

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

        // Create Ray Tracing resources

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raytrace Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "renderer/shaders/ray_trace.wgsl"
            ))),
        });

        let tlas_bgroup_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BindGroup Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raytrace Layout"),
            bind_group_layouts: &[
                &bgroup_camera::WgpuBindGroup0::get_bind_group_layout(&device),
                &tlas_bgroup_layout,
            ],
            push_constant_ranges: &[],
        });
        let raytrace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raytrace Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let tlas = device.create_tlas(&wgpu::wgt::CreateTlasDescriptor {
            label: Some("Default Scene TLAS"),
            max_instances: 0,
            flags: wgpu::AccelerationStructureFlags::empty(),
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let raytrace_default_bgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene TLAS Default BindGroup"),
            layout: &tlas_bgroup_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: Some("Build default TLAS")});
        encoder.build_acceleration_structures([], &[tlas]);
        queue.submit([encoder.finish()]);

        Self {
            user_camera,
            camera_buf,
            camera_bgroup,

            skybox_bgroup,
            skybox_pipeline,

            asset_rx,
            asset_tx,

            asset_list: asset_list_rx,

            raytrace_pipeline,
            raytrace_default_bgroup,
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

        rpass.set_bind_group(1, Some(&self.raytrace_default_bgroup), &[]);
        rpass.set_pipeline(&self.raytrace_pipeline);
        rpass.draw(0..3, 0..1);

        // self.skybox_bgroup.set(rpass);
        // rpass.set_pipeline(&self.skybox_pipeline);
        // rpass.draw(0..3, 0..1);
    }

    pub fn run_ui(&mut self, ctx: &egui::Context, render_state: &egui_wgpu::RenderState) {
        profile_function!();

        if !ctx.wants_keyboard_input() && !ctx.wants_pointer_input() {
            ctx.input(|input| {
                self.user_camera.update(input);
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Asset", |ui| {
                    self.show_scene_menu(&render_state.device, render_state.target_format, ui);
                });

                ui.menu_button("Camera", |ui| self.user_camera.run_ui(ui));

                ui.menu_button("Info", |ui| {
                    ui.menu_button("Adapter", |ui| {
                        let info = render_state.adapter.get_info();
                        ui.label(format!("name: {}", info.name));
                        ui.label(format!("backend: {}", info.backend));
                        ui.label(format!("driver: {}", info.driver));
                        ui.label(format!("driver info: {}", info.driver_info));
                        ui.label(format!("type: {:?}", info.device_type));
                    });
                });
            });
        });
    }

    fn show_scene_menu(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        ui: &mut egui::Ui,
    ) {
        let model_list = self.asset_list.borrow();
        if !model_list.is_empty() {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for model in model_list.iter() {
                    ui.menu_button(&model.label, |ui| {
                        for (variant, file) in &model.variants {
                            if ui.button(variant).clicked() {
                                let asset_tx = self.asset_tx.clone();
                                let device = device.clone();
                                let url = Url::parse(ASSETS_BASE_URL)
                                    .unwrap()
                                    .join(&format!("{}/{}/{}", &model.name, variant, file))
                                    .unwrap();
                                // crate::spawn(async move {
                                //     let scene = gltf_loader::load_asset(url, &device, color_format)
                                //         .await
                                //         .unwrap();
                                //     let _ = asset_tx.send(Some(scene));
                                // });
                            }
                        }
                    });
                }
            });
        }
    }
}
