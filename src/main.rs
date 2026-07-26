mod datatype;
mod drawline;
mod drawtriangle;
mod model;
mod renderpipeline;
mod tgaimage;
mod framebuffer;

use minifb::{Key, Window, WindowOptions};
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::path::Path;
use std::time::Instant;

use model::Model;
use renderpipeline::{RenderPipleline, Uniforms, VertexInput, Varying};

use crate::drawline::WHITE;
use crate::framebuffer::FrameBuffer;
use crate::renderpipeline::{lookat, projection};
use crate::tgaimage::TGAColor;

const DEBUG_FPS: bool = false;

fn main() {
    run();
}

fn run() {
    let vertexs_data = vec![
        load_model("assert/african_head/african_head.obj"),
        load_model("assert/african_head/african_head_eye_inner.obj"),
        load_model("assert/african_head/african_head_eye_outer.obj"),
    ];

    let width = 800;
    let height = 800;

    // 1. 创建帧缓冲 & 设置背景色
    let mut framebuffer = FrameBuffer::new(width, height);
    framebuffer.clear_color(&TGAColor {
        r: 30.0 / 255.0,
        g: 30.0 / 255.0,
        b: 30.0 / 255.0,
        a: 1.0,
    });

    // 2. 设置相机 / 投影变换
    let model_mat = Mat4::IDENTITY;
    let eye = Vec3::new(1.0, 0.0, 2.5);
    let center = Vec3::ZERO;
    let up = Vec3::Y;
    let view_mat = lookat(&eye, &center, &up);
    let proj_mat = projection(
        renderpipeline::ProjectionMode::PERSPECTIVE,
        std::f32::consts::FRAC_PI_4,
        Vec2 {
            x: width as f32,
            y: height as f32,
        },
        0.1,
        10.0,
    );

    let model_view = view_mat * model_mat;
    let model_view_proj = proj_mat * model_view;
    let normal_matrix = Mat3::from_mat4(model_mat.inverse().transpose());

    let uniforms = Uniforms {
        model: model_mat,
        view: view_mat,
        projection: proj_mat,
        model_view,
        model_view_proj,
        normal_matrix,
        light_dir: Vec3::new(-1.0, 1.0, 1.0).normalize(),
        view_dir: (eye - center),
        ambient_color: Vec3::new(0.5, 0.5, 0.5),
        diffuse_color: Vec3::new(0.7, 0.7, 0.7),
        specular_color: Vec3::new(0.3, 0.3, 0.3),
        diffuse_tex: None,
        normal_tex: None,
        specular_tex: None,
        glossiness_tex: None,
    };

    // 3. 运行渲染管线
    let mut pipeline = RenderPipleline::new(&mut framebuffer);
    pipeline.set_flat_normal(false);
    pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);

    for vertexs_opt in vertexs_data {
        if let Some(verts) = vertexs_opt {
            pipeline.add_data(verts);
            pipeline.set_uniforms(&uniforms);
            pipeline.draw();
        }
    }

    // 4. 保存为 TGA 文件
    framebuffer.save_to_image("output_render.tga");

    // 5. 打开 minifb 窗口显示渲染结果（不限帧率）
    let mut window = Window::new(
        "Tiny Renderer",
        width,
        height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{}", e));

    window.set_target_fps(0); // 不限帧率

    let mut fps_timer = Instant::now();
    let mut frame_count = 0u32;
    let mut fps = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(framebuffer.as_buffer(), width, height)
            .unwrap();

        frame_count += 1;
        let elapsed = fps_timer.elapsed();
        if elapsed.as_secs_f64() >= 1.0 {
            fps = frame_count;
            frame_count = 0;
            fps_timer = Instant::now();

            // 主方案：在窗口标题显示 FPS
            window.set_title(&format!("Tiny Renderer - {} FPS", fps));

            // 备用方案：debug 开关输出 FPS 到控制台
            if DEBUG_FPS {
                println!("[DEBUG] FPS: {}", fps);
            }
        }
    }
}

fn load_model(path: &str) -> Option<Vec<VertexInput>> {
    let model: Model = Model::new(Path::new(path));
    println!(
        "模型加载成功: {} 顶点, {} 面",
        model.verts().len() - 1,
        model.faces().len() - 1,
    );

    let mut vertices: Vec<VertexInput> = Vec::new();
    for face in model.faces() {
        if face.len() == 3 {
            for idx in face {
                let pos = model.verts()[idx[0] as usize];
                let normal = model.vert_normals()[idx[2] as usize];
                let texcoord = {
                    let vt = model.texture_verts()[idx[1] as usize];
                    Vec2::new(vt.x, vt.y)
                };
                vertices.push(VertexInput {
                    pos,
                    varyings: vec![
                        Varying::Color(WHITE),
                        Varying::Vec3(normal),
                        Varying::Vec2(texcoord),
                    ],
                });
            }
        }
    }
    println!(
        "组装 {} 个顶点输入 ({} 个三角形)",
        vertices.len(),
        vertices.len() / 3
    );
    Some(vertices)
}
