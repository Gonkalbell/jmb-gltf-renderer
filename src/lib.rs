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

use eframe::{
    egui::ahash::HashMap, wgpu,
};
use reqwest::Url;
use serde::Deserialize;
use tokio::sync::watch::{Receiver, Sender};

use crate::{
    asset::{Asset, LoadingProgress},
    camera::ArcBallCamera,
    skybox::Skybox,
};

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
    pub camera: ArcBallCamera,

    pub skybox: Skybox,

    pub asset_rx: Receiver<Option<Asset>>,
    pub asset_tx: Sender<Option<Asset>>,

    pub asset_list: Receiver<Vec<ModelLinkInfo>>,
    pub loading_progress: Arc<Mutex<LoadingProgress>>,
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
}

#[derive(Debug, Deserialize)]
pub struct ModelLinkInfo {
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
