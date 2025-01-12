use super::{
    scene, AttribBuffer, Mesh, Node, NodeBindGroup, NodeBindGroupEntries,
    NodeBindGroupEntriesParams, Primitive, PrimitiveIndexData, DEPTH_FORMAT,
};
use glam::{Mat3, Mat4};
use gltf::mesh::Mode;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(crate) fn generate_nodes(doc: &gltf::Document, device: &wgpu::Device) -> Vec<Node> {
    // Get world transforms
    let mut nodes_to_visit = Vec::new();
    for scene in doc.scenes() {
        nodes_to_visit.extend(scene.nodes().map(|n| (n, Mat4::IDENTITY)));
    }
    let mut world_transforms = vec![Mat4::IDENTITY; doc.nodes().len()];
    while let Some((node, parent_transform)) = nodes_to_visit.pop() {
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_transform = parent_transform * transform;
        world_transforms[node.index()] = world_transform;
        nodes_to_visit.extend(node.children().map(|n| (n, world_transform)));
    }
    let nodes = doc
        .nodes()
        .zip(world_transforms.iter())
        .filter_map(|(node, &transform)| {
            node.mesh().map(|mesh| {
                let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: node.name(),
                    contents: bytemuck::bytes_of(&scene::Node {
                        transform,
                        normal_transform: Mat4::from_mat3(
                            Mat3::from_mat4(transform).inverse().transpose(),
                        ),
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bgroup = NodeBindGroup::from_bindings(
                    device,
                    NodeBindGroupEntries::new(NodeBindGroupEntriesParams {
                        res_node: node_buf.as_entire_buffer_binding(),
                    }),
                );
                Node {
                    bgroup,
                    mesh_index: mesh.index(),
                }
            })
        })
        .collect();
    nodes
}

pub(crate) fn generate_meshes(
    device: &wgpu::Device,
    doc: gltf::Document,
    buffers: Vec<Arc<wgpu::Buffer>>,
    color_format: wgpu::TextureFormat,
) -> Vec<Mesh> {
    let shader = scene::create_shader_module_embed_source(device);

    doc.meshes()
        .map(|doc_mesh| {
            let primitives = doc_mesh
                .primitives()
                .map(|doc_primitive| {
                    let vertex_count = doc_primitive
                        .attributes()
                        .next()
                        .expect("There should be at least one attribute for each primitive")
                        .1
                        .count();
                    let (attrib_layouts, attrib_buffers): (Vec<_>, Vec<_>) = doc_primitive
                        .attributes()
                        .filter_map(|(semantic, accessor)| {
                            let shader_location = match semantic {
                                gltf::Semantic::Positions => 0,
                                gltf::Semantic::Normals => 1,
                                _ => return None,
                            };

                            let buffer_view =
                                accessor.view().expect("Accessor should have a buffer view");
                            let format = get_vertex_format(&accessor);
                            let stride = buffer_view
                                .stride()
                                .map(|s| s as u64)
                                .unwrap_or(format.size());
                            let (buf_offset, attrib_offset) = if accessor.offset()
                                >= stride as _
                                || stride > device.limits().max_vertex_buffer_array_stride as _
                            {
                                (accessor.offset(), 0)
                            } else {
                                (0, accessor.offset())
                            };
                            Some((
                                (
                                    stride,
                                    wgpu::VertexAttribute {
                                        format,
                                        offset: attrib_offset as _,
                                        shader_location,
                                    },
                                ),
                                AttribBuffer {
                                    buffer: buffers[buffer_view.index()].clone(),
                                    offset: buf_offset as _,
                                },
                            ))
                        })
                        .unzip();

                    let attrib_buffer_layouts: Vec<_> = attrib_layouts
                        .iter()
                        .map(|(array_stride, attributes)| wgpu::VertexBufferLayout {
                            array_stride: *array_stride,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: std::slice::from_ref(attributes),
                        })
                        .collect();

                    let pipeline =
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: doc_mesh.name(),
                            layout: Some(&scene::create_pipeline_layout(device)),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some(scene::ENTRY_VS_SCENE),
                                compilation_options: Default::default(),
                                buffers: &attrib_buffer_layouts,
                            },
                            fragment: Some(scene::fragment_state(
                                &shader,
                                &scene::fs_scene_entry([Some(color_format.into())]),
                            )),
                            primitive: wgpu::PrimitiveState {
                                topology: match doc_primitive.mode() {
                                    Mode::Points => wgpu::PrimitiveTopology::PointList,
                                    Mode::Lines => wgpu::PrimitiveTopology::LineList,
                                    Mode::LineStrip => wgpu::PrimitiveTopology::LineStrip,
                                    Mode::Triangles => wgpu::PrimitiveTopology::TriangleList,
                                    Mode::TriangleStrip => {
                                        wgpu::PrimitiveTopology::TriangleStrip
                                    }
                                    mode => unimplemented!("format {:?} not supported", mode),
                                },
                                cull_mode: Some(wgpu::Face::Back),
                                front_face: wgpu::FrontFace::Ccw,
                                ..Default::default()
                            },
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: DEPTH_FORMAT,
                                depth_write_enabled: true,
                                depth_compare: wgpu::CompareFunction::Less,
                                stencil: wgpu::StencilState::default(),
                                bias: wgpu::DepthBiasState::default(),
                            }),
                            multisample: wgpu::MultisampleState::default(),
                            multiview: None,
                            cache: None,
                        });

                    let mut draw_count = vertex_count as u32;
                    let index_data = doc_primitive.indices().map(|indices| {
                        use gltf::accessor::DataType;
                        draw_count = indices.count() as _;
                        PrimitiveIndexData {
                            buffer: buffers[indices.view().unwrap().index()].clone(),
                            format: match indices.data_type() {
                                DataType::U16 => wgpu::IndexFormat::Uint16,
                                DataType::U32 => wgpu::IndexFormat::Uint32,
                                t => unimplemented!("Index type {:?} is not supported", t),
                            },
                            offset: indices.offset() as _,
                        }
                    });

                    Primitive {
                        pipeline,
                        attrib_buffers,
                        draw_count,
                        index_data,
                    }
                })
                .collect();
            Mesh { primitives }
        })
        .collect()
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
