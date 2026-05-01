use miette::{IntoDiagnostic, Result};
use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

// src/build.rs
fn main() -> Result<()> {
    WgslBindgenOptionBuilder::default()
        .workspace_root("src/shaders")
        .add_entry_point("src/shaders/scene.wgsl")
        .add_entry_point("src/shaders/skybox.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap)
        .derive_serde(true)
        .output("src/shaders.rs")
        .build()?
        .generate()
        .into_diagnostic()
}
