#![warn(clippy::all)]

mod app;
mod asset;
mod camera;
mod skybox;

#[allow(clippy::all)]
mod shaders;

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

const ASSETS_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub mod bind_groups {
    use super::shaders::*;

    pub type Camera = bgroup_camera::WgpuBindGroup0;
    pub type CameraEntries<'a> = bgroup_camera::WgpuBindGroup0Entries<'a>;
    pub type CameraEntriesParams<'a> = bgroup_camera::WgpuBindGroup0EntriesParams<'a>;

    pub type Material = scene::WgpuBindGroup1;
    pub type MaterialEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
    pub type MaterialEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

    pub type Instance = scene::WgpuBindGroup2;
    pub type InstanceEntries<'a> = scene::WgpuBindGroup2Entries<'a>;
    pub type InstanceEntriesParams<'a> = scene::WgpuBindGroup2EntriesParams<'a>;

    pub type Skybox = skybox::WgpuBindGroup1;
    pub type SkyboxEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
    pub type SkyboxEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;
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
