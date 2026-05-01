use super::{
    DEPTH_FORMAT, OwnedBufferSlice, bind_groups, shaders::scene::{self, Instance, VertexInput},
};

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Write,
    ops::Range,
    sync::{Arc, Mutex},
};

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use gltf::mesh::Mode;
use image::DynamicImage;
use reqwest::Url;
use wgpu::util::DeviceExt;
use wgpu::wgt::TextureViewDescriptor;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct OwnedVertexBufferLayout {
    array_stride: wgpu::BufferAddress,
    attribute: wgpu::VertexAttribute,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineCacheKey {
    attributes: Vec<OwnedVertexBufferLayout>,
    primitive_state: wgpu::PrimitiveState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asset {
    info: String,
    batches: Vec<RenderBatch>,
    instance_bgroup: bind_groups::Instance,
}

impl Asset {
    pub fn info(&self) -> &str {
        &self.info
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass<'_>) {
        self.instance_bgroup.set(rpass);

        for batch in self.batches.iter() {
            rpass.set_pipeline(&batch.pipeline);

            for primitive in batch.mesh_primitives.iter() {
                for (i, attrib) in primitive.attrib_buffers.iter().enumerate() {
                    rpass.set_vertex_buffer(i as _, attrib.as_slice());
                }

                if let Some(index_data) = &primitive.index_data {
                    rpass.set_index_buffer(index_data.buffer_slice.as_slice(), index_data.format);
                }

                if primitive.index_data.is_none() {
                    rpass.draw(0..primitive.draw_count, primitive.instances.clone());
                } else {
                    rpass.draw_indexed(0..primitive.draw_count, 0, primitive.instances.clone());
                }
            }
        }
    }
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

pub struct LoadingProgress {
    pub loaded: usize,
    pub total: usize,
}

pub async fn load_asset(
    url: Url,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
    loading_progress: Arc<Mutex<LoadingProgress>>,
) -> anyhow::Result<Asset> {
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded = 0;
        loading_progress.total = 1;
    }
    let gltf_file = gltf::Gltf::from_slice(&request_data(&url, loading_progress.clone()).await?)?;
    let doc = gltf_file.document;

    let mut buffer_contents = futures::future::try_join_all(doc.buffers().map(|doc_buffer| {
        use gltf::buffer::Source;
        let url = url.clone();
        let source = doc_buffer.source().clone();
        let loading_progress = loading_progress.clone();
        async move {
            let contents = match source {
                Source::Bin => Vec::new(),
                Source::Uri(uri) => request_data(&url.join(uri)?, loading_progress).await?,
            };
            Ok::<_, anyhow::Error>(contents)
        }
    }))
    .await?;

    if let Some(blob) = gltf_file.blob {
        buffer_contents[0] = blob
    }

    let buffers: Vec<_> = doc
        .buffers()
        .zip(buffer_contents.iter())
        .map(|(doc_buffer, contents)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: doc_buffer.name(),
                contents,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::INDEX,
            })
        })
        .collect();

    let buffer_slices: Vec<_> = doc
        .views()
        .map(|doc_view: gltf::buffer::View| {
            buffers[doc_view.buffer().index()].slice(
                doc_view.offset() as wgpu::BufferAddress
                    ..(doc_view.offset() + doc_view.length()) as wgpu::BufferAddress,
            )
        })
        .collect();

    let _textures = futures::future::try_join_all(doc.images().map(|doc_image| {
        use gltf::image::Source;
        let source = doc_image.source();
        async {
            let contents = match source {
                Source::View { view, .. } => {
                    let parent_buffer_data = &buffer_contents[view.buffer().index()];
                    let begin = view.offset();
                    let end = begin + view.length();
                    let contents = &parent_buffer_data[begin..end];
                    Cow::Borrowed(contents)
                }
                Source::Uri { uri, .. } => {
                    let data = request_data(&url.join(uri)?, loading_progress.clone()).await?;
                    Cow::Owned(data)
                }
            };

            let dynamic_image = image::load_from_memory(&contents)?;

            // wgpu doesn't support 3 channel types, so I need to convert these
            let dynamic_image = match &dynamic_image {
                DynamicImage::ImageRgb8(_) => DynamicImage::ImageRgba8(dynamic_image.to_rgba8()),
                DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgba16(dynamic_image.to_rgba16()),
                DynamicImage::ImageRgba32F(_) => {
                    DynamicImage::ImageRgba32F(dynamic_image.to_rgba32f())
                }
                _ => dynamic_image,
            };
            let format = match &dynamic_image {
                DynamicImage::ImageLuma8(_) => wgpu::TextureFormat::R8Unorm,
                DynamicImage::ImageLumaA8(_) => wgpu::TextureFormat::Rg8Unorm,
                DynamicImage::ImageRgba8(_) => wgpu::TextureFormat::Rgba8Unorm,
                DynamicImage::ImageLuma16(_) => wgpu::TextureFormat::R16Unorm,
                DynamicImage::ImageLumaA16(_) => wgpu::TextureFormat::Rg16Unorm,
                DynamicImage::ImageRgba16(_) => wgpu::TextureFormat::Rgba16Unorm,
                DynamicImage::ImageRgb32F(_) => wgpu::TextureFormat::Rgba32Float,
                other_format => {
                    return Err(anyhow::anyhow!("Unsupported format {:?}", other_format));
                }
            };
            let size: wgpu::Extent3d = wgpu::Extent3d {
                width: dynamic_image.width(),
                height: dynamic_image.height(),
                ..Default::default()
            };
            let mip_level_count = size.width.min(size.height).ilog2().max(1);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[format.add_srgb_suffix(), format.remove_srgb_suffix()],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfoBase {
                    texture: &texture,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default(),
                },
                dynamic_image.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: Default::default(),
                    bytes_per_row: format.block_copy_size(None).map(|b| b * size.width),
                    rows_per_image: Default::default(),
                },
                size,
            );

            let blitter = wgpu::util::TextureBlitterBuilder::new(device, format)
                .sample_type(wgpu::FilterMode::Linear)
                .build();

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blitting"),
            });
            for base_mip_level in 1..mip_level_count {
                blitter.copy(
                    device,
                    &mut encoder,
                    &texture.create_view(&TextureViewDescriptor {
                        base_mip_level: base_mip_level - 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                    &texture.create_view(&TextureViewDescriptor {
                        base_mip_level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }),
                );
            }

            queue.submit([encoder.finish()]);

            Ok::<_, anyhow::Error>(dynamic_image)
        }
    }))
    .await?;

    let samplers: Vec<_> = doc
        .samplers()
        .map(|doc_sampler| {
            use gltf::texture::{MagFilter, MinFilter, WrappingMode};

            device.create_sampler(&wgpu::SamplerDescriptor {
                label: doc_sampler.name(),
                address_mode_u: match doc_sampler.wrap_s() {
                    WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                address_mode_v: match doc_sampler.wrap_s() {
                    WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                mag_filter: match doc_sampler.mag_filter() {
                    Some(MagFilter::Nearest) => wgpu::FilterMode::Nearest,
                    Some(MagFilter::Linear) | None => wgpu::FilterMode::Linear,
                },
                min_filter: match doc_sampler.min_filter() {
                    Some(
                        MinFilter::Nearest
                        | MinFilter::NearestMipmapLinear
                        | MinFilter::NearestMipmapNearest,
                    ) => wgpu::FilterMode::Nearest,
                    None
                    | Some(
                        MinFilter::Linear
                        | MinFilter::LinearMipmapLinear
                        | MinFilter::LinearMipmapNearest,
                    ) => wgpu::FilterMode::Linear,
                },
                mipmap_filter: match doc_sampler.min_filter() {
                    Some(
                        MinFilter::Nearest
                        | MinFilter::LinearMipmapNearest
                        | MinFilter::NearestMipmapNearest,
                    ) => wgpu::MipmapFilterMode::Nearest,
                    _ => wgpu::MipmapFilterMode::Linear,
                },
                ..Default::default()
            })
        })
        .collect();
    let _ = samplers;

    let (instance_bgroup, mesh_instances) = generate_nodes(device, &doc);

    let batches = generate_meshes(device, &doc, color_format, &buffer_slices, &mesh_instances);

    let mut asset_info = String::new();
    let json_asset = &doc.as_json().asset;
    writeln!(&mut asset_info, "version: {}", json_asset.version)?;
    if let Some(min_version) = &json_asset.min_version {
        writeln!(&mut asset_info, "min_version: {}", min_version)?;
    }
    if let Some(copyright) = &json_asset.copyright {
        writeln!(&mut asset_info, "copyright: {}", copyright)?;
    }
    if let Some(generator) = &json_asset.generator {
        writeln!(&mut asset_info, "generator: {}", generator)?;
    }

    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded = loading_progress.total;
    }

    log::info!("finished loading {}", &url);
    Ok(Asset {
        info: asset_info,
        batches,
        instance_bgroup,
    })
}

async fn request_data(
    url: &Url,
    loading_progress: Arc<Mutex<LoadingProgress>>,
) -> anyhow::Result<Vec<u8>> {
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.total += 1;
    }
    let data = match url.scheme() {
        "http" | "https" => reqwest::get(url.clone()).await?.bytes().await?.to_vec(),
        #[cfg(not(target_family = "wasm"))]
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL"))?;
            tokio::fs::read(path).await?
        }
        "data" => {
            let data_url =
                data_url::DataUrl::process(url.as_str()).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let (data, _) = data_url
                .decode_to_vec()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            data
        }
        other => anyhow::bail!("Unsupported scheme: {}", other),
    };
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded += 1;
    }
    Ok(data)
}

#[derive(Hash, Eq, PartialEq, PartialOrd, Ord)]
struct MeshIndex(usize);

fn generate_nodes(
    device: &wgpu::Device,
    doc: &gltf::Document,
) -> (bind_groups::Instance, HashMap<MeshIndex, Range<u32>>) {
    // Get world transforms
    let mut nodes_to_visit = Vec::new();
    for doc_scene in doc.scenes() {
        nodes_to_visit.extend(doc_scene.nodes().map(|n| (n, Mat4::IDENTITY)));
    }
    let mut world_transforms = vec![Mat4::IDENTITY; doc.nodes().len()];
    while let Some((node, parent_transform)) = nodes_to_visit.pop() {
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_transform = parent_transform * transform;
        world_transforms[node.index()] = world_transform;
        nodes_to_visit.extend(node.children().map(|n| (n, world_transform)));
    }
    let (bbox_min, bbox_max) = doc
        .nodes()
        .zip(world_transforms.iter())
        .filter_map(|(node, transform)| node.mesh().map(|m| (m, transform)))
        .flat_map(|(mesh, transform)| mesh.primitives().map(|p| (p.bounding_box(), *transform)))
        .fold((Vec3::MAX, Vec3::MIN), |(min, max), (bbox, transform)| {
            (
                min.min(transform.transform_point3(bbox.min.into())),
                max.max(transform.transform_point3(bbox.max.into())),
            )
        });

    let center = (bbox_min + bbox_max) * 0.5;
    let extent = bbox_max - bbox_min;
    let max_extent = extent.max_element();
    let scale_factor = max_extent.recip();

    let inv_bounding_box_matrix = Mat4::from_scale_rotation_translation(
        Vec3::splat(scale_factor),
        Quat::IDENTITY,
        scale_factor * -center,
    );

    for transform in world_transforms.iter_mut() {
        *transform = inv_bounding_box_matrix * *transform;
    }

    let mut mesh_instances: HashMap<MeshIndex, Vec<Instance>> = HashMap::new();
    for (doc_node, &transform) in doc.nodes().zip(world_transforms.iter()) {
        if let Some(doc_mesh) = doc_node.mesh() {
            mesh_instances
                .entry(MeshIndex(doc_mesh.index()))
                .or_default()
                .push(Instance {
                    local_to_world: transform,
                    normal_local_to_world: Mat4::from_mat3(
                        Mat3::from_mat4(transform).inverse().transpose(),
                    ),
                });
        }
    }

    let mut all_instance_data = Vec::new();
    let mut mesh_instance_ranges: HashMap<MeshIndex, Range<u32>> = HashMap::new();
    for (mesh_index, instances) in mesh_instances {
        let start = all_instance_data.len() as u32;
        all_instance_data.extend_from_slice(&instances);
        let end = all_instance_data.len() as u32;
        mesh_instance_ranges.insert(mesh_index, start..end);
    }

    let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instances"),
        contents: bytemuck::cast_slice(&all_instance_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let instance_bgroup = bind_groups::Instance::from_bindings(
        device,
        bind_groups::InstanceEntries::new(bind_groups::InstanceEntriesParams {
            res_instances: instance_buf.as_entire_buffer_binding(),
        }),
    );

    (instance_bgroup, mesh_instance_ranges)
}

fn generate_meshes(
    device: &wgpu::Device,
    doc: &gltf::Document,
    color_format: wgpu::TextureFormat,
    buffer_slices: &[wgpu::BufferSlice],
    mesh_instances: &HashMap<MeshIndex, Range<u32>>,
) -> Vec<RenderBatch> {
    let shader = scene::create_shader_module_embed_source(device);
    let mut batches = HashMap::new();

    let default_vertex_input = VertexInput {
        position: Default::default(),
        normal: Vec3::ZERO,
        tangent: Vec4::ZERO,
        texcoord_0: Default::default(),
        texcoord_1: Default::default(),
        color_0: Vec4::ONE,
        color_1: Vec4::ONE,
    };
    let default_vertex_buf: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Default Vertex Input"),
            contents: bytemuck::bytes_of(&default_vertex_input),
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
        });
    let default_buf_and_layout_iter = VertexInput::VERTEX_ATTRIBUTES.into_iter().map(|attrib| {
        (
            OwnedBufferSlice::from_slice(&default_vertex_buf.slice(..)),
            OwnedVertexBufferLayout {
                array_stride: 0,
                attribute: attrib,
            },
        )
    });
    let semantics = [
        gltf::Semantic::Positions,
        gltf::Semantic::Normals,
        gltf::Semantic::Tangents,
        gltf::Semantic::TexCoords(0),
        gltf::Semantic::TexCoords(1),
        gltf::Semantic::Colors(0),
        gltf::Semantic::Colors(1),
    ];

    let default_attributes: HashMap<gltf::Semantic, (OwnedBufferSlice, OwnedVertexBufferLayout)> =
        HashMap::from_iter(semantics.into_iter().zip(default_buf_and_layout_iter));

    for doc_mesh in doc.meshes() {
        for doc_primitive in doc_mesh.primitives() {
            let vertex_count = doc_primitive
                .attributes()
                .next()
                .expect("There should be at least one attribute for each primitive")
                .1
                .count();

            let mut attributes = default_attributes.clone();
            for (semantic, accessor) in doc_primitive.attributes() {
                let shader_location =
                    if let Some((_, attrib_layout)) = default_attributes.get(&semantic) {
                        attrib_layout.attribute.shader_location
                    } else {
                        continue;
                    };
                let view = accessor.view().unwrap();
                let format = get_vertex_format(&accessor);
                let accessor_end = accessor.offset() as wgpu::BufferAddress + format.size();
                let array_stride = view.stride().map(|s| s as _).unwrap_or(format.size());
                let buf_slice = buffer_slices[view.index()];

                let (offset, buf_slice) = if accessor_end <= array_stride {
                    (accessor.offset() as _, buf_slice)
                } else {
                    // While normally I can have one wgpu::BufferSlice per gltf::View, some assets use accessors to
                    // essentially act as a new view rather than an offset into a "stride" sized block. So for these
                    // cases, I need to treat these accessors as if they have a completely new wgpu::BufferSlice
                    (
                        0,
                        buf_slice.slice(accessor.offset() as wgpu::BufferAddress..),
                    )
                };

                let owned_slice = OwnedBufferSlice::from_slice(&buf_slice);
                let owned_layout = OwnedVertexBufferLayout {
                    array_stride,
                    attribute: wgpu::VertexAttribute {
                        format,
                        offset,
                        shader_location,
                    },
                };

                attributes.insert(semantic, (owned_slice, owned_layout));
            }

            let (attrib_buffers, attrib_layouts): (Vec<_>, Vec<_>) =
                attributes.into_values().unzip();

            let primitive_state = wgpu::PrimitiveState {
                topology: match doc_primitive.mode() {
                    Mode::Points => wgpu::PrimitiveTopology::PointList,
                    Mode::Lines => wgpu::PrimitiveTopology::LineList,
                    Mode::LineStrip => wgpu::PrimitiveTopology::LineStrip,
                    Mode::Triangles => wgpu::PrimitiveTopology::TriangleList,
                    Mode::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                    mode => unimplemented!("format {:?} not supported", mode),
                },
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            };
            let key = PipelineCacheKey {
                attributes: attrib_layouts,
                primitive_state,
            };

            let mut draw_count = vertex_count as u32;
            let index_data = doc_primitive.indices().map(|indices| {
                use gltf::accessor::DataType;
                draw_count = indices.count() as _;
                PrimitiveIndexData {
                    buffer_slice: OwnedBufferSlice::from_slice(
                        &buffer_slices[indices.view().unwrap().index()]
                            .slice(indices.offset() as wgpu::BufferAddress..),
                    ),
                    format: match indices.data_type() {
                        DataType::U16 => wgpu::IndexFormat::Uint16,
                        DataType::U32 => wgpu::IndexFormat::Uint32,
                        t => unimplemented!("Index type {:?} is not supported", t),
                    },
                }
            });

            let batch = batches.entry(key).or_insert_with_key(|key| {
                create_render_batch(device, color_format, &shader, None, key)
            });
            let instances = mesh_instances
                .get(&MeshIndex(doc_mesh.index()))
                .unwrap()
                .clone();

            batch.mesh_primitives.push(Primitive {
                attrib_buffers,
                draw_count,
                index_data,
                instances,
            });
        }
    }

    batches.into_values().collect()
}

fn create_render_batch(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
    label: Option<&str>,
    key: &PipelineCacheKey,
) -> RenderBatch {
    let attrib_buffer_layouts: Vec<_> = key
        .attributes
        .iter()
        .map(
            |OwnedVertexBufferLayout {
                 array_stride,
                 attribute,
             }| wgpu::VertexBufferLayout {
                array_stride: *array_stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: std::slice::from_ref(attribute),
            },
        )
        .collect();

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(&scene::create_pipeline_layout(device)),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(scene::ENTRY_VS_SCENE),
            compilation_options: Default::default(),
            buffers: &attrib_buffer_layouts,
        },
        fragment: Some(scene::fragment_state(
            shader,
            &scene::fs_scene_entry([Some(color_format.into())]),
        )),
        primitive: key.primitive_state,
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    RenderBatch {
        pipeline,
        mesh_primitives: Vec::new(),
    }
}

fn get_vertex_format(accessor: &gltf::Accessor) -> wgpu::VertexFormat {
    use gltf::accessor::{DataType, Dimensions};
    match (
        accessor.normalized(),
        accessor.data_type(),
        accessor.dimensions(),
    ) {
        (true, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Snorm8x2,
        (true, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Snorm8x4,
        (true, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Unorm8x2,
        (true, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Unorm8x4,
        (true, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Snorm16x2,
        (true, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Snorm16x4,
        (true, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Unorm16x2,
        (true, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Unorm16x4,
        (false, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Sint8x2,
        (false, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Sint8x4,
        (false, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Uint8x2,
        (false, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Uint8x4,
        (false, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Sint16x2,
        (false, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Sint16x4,
        (false, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Uint16x2,
        (false, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Uint16x4,
        (false, DataType::U32, Dimensions::Scalar) => wgpu::VertexFormat::Uint32,
        (false, DataType::U32, Dimensions::Vec2) => wgpu::VertexFormat::Uint32x2,
        (false, DataType::U32, Dimensions::Vec3) => wgpu::VertexFormat::Uint32x3,
        (false, DataType::U32, Dimensions::Vec4) => wgpu::VertexFormat::Uint32x4,
        (_, DataType::F32, Dimensions::Scalar) => wgpu::VertexFormat::Float32,
        (_, DataType::F32, Dimensions::Vec2) => wgpu::VertexFormat::Float32x2,
        (_, DataType::F32, Dimensions::Vec3) => wgpu::VertexFormat::Float32x3,
        (_, DataType::F32, Dimensions::Vec4) => wgpu::VertexFormat::Float32x4,
        _ => unimplemented!(),
    }
}
