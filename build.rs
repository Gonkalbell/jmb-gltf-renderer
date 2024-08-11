use miette::{IntoDiagnostic, Result};
use wgsl_bindgen::{WgslTypeSerializeStrategy, WgslBindgenOptionBuilder, GlamWgslTypeMap};

// src/build.rs
fn main() -> Result<()> {
    WgslBindgenOptionBuilder::default()
        .workspace_root("src/renderer/shaders")
        .add_entry_point("src/renderer/shaders/cube.wgsl")
        .add_entry_point("src/renderer/shaders/skybox.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap)
        .derive_serde(true)
        .output("src/renderer/shader.rs")
        .build()?
        .generate()
        .into_diagnostic()
}