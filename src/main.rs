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
    // 一次性加载所有模型
    let vertexs_data: Vec<Vec<VertexInput>> = vec![
        load_model("assert/african_head/african_head.obj"),
        load_model("assert/african_head/african_head_eye_inner.obj"),
        load_model("assert/african_head/african_head_eye_outer.obj"),
    ]
    .into_iter()
    .flatten()
    .collect();

    let width = 800;
    let height = 800;
    let bg_color = TGAColor {
        r: 30.0 / 255.0,
        g: 30.0 / 255.0,
        b: 30.0 / 255.0,
        a: 1.0,
    };

    let mut framebuffer = FrameBuffer::new(width, height);
    let mut pipeline = RenderPipleline::new(&mut framebuffer);
    pipeline.set_flat_normal(false);
    pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);

    // 投影矩阵（不变）
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

    // 打开 minifb 窗口（不限帧率）
    let mut window = Window::new("Tiny Renderer", width, height, WindowOptions::default())
        .unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0);

    // 相机状态：球面坐标
    let mut yaw = 0.0f32;
    let mut pitch = 0.0f32;
    let radius = 2.5f32;
    let center = Vec3::ZERO;
    let up = Vec3::Y;
    let model_mat = Mat4::IDENTITY;
    const ROTATE_SPEED: f32 = 0.05;
    let light_dir = Vec3::new(-1.0, 1.0, 1.0).normalize();

    let mut fps_timer = Instant::now();
    let mut frame_count = 0u32;
    let mut fps = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ----- 处理输入：上下左右旋转相机 -----
        if window.is_key_down(Key::Left) {
            yaw -= ROTATE_SPEED;
        }
        if window.is_key_down(Key::Right) {
            yaw += ROTATE_SPEED;
        }
        if window.is_key_down(Key::Up) {
            pitch = (pitch + ROTATE_SPEED).min(1.5);
        }
        if window.is_key_down(Key::Down) {
            pitch = (pitch - ROTATE_SPEED).max(-1.5);
        }

        // ----- 相机位置（球面 → 笛卡尔）-----
        let eye = radius * Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        );

        let view_mat = lookat(&eye, &center, &up);
        let model_view = view_mat * model_mat;
        let model_view_proj = proj_mat * model_view;
        let normal_matrix = Mat3::from_mat4(model_mat.inverse().transpose());

        // view_dir: 从表面指向相机的方向 → 用于 Blinn-Phong 半向量
        let view_dir = (eye - center).normalize();

        let uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view,
            model_view_proj,
            normal_matrix,
            light_dir,
            view_dir,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: None,
            normal_tex: None,
            specular_tex: None,
            glossiness_tex: None,
        };

        // ----- 渲染（复用 pipeline，不复分配）-----
        pipeline.clear_buffer(&bg_color);
        pipeline.begin_frame();

        for verts in &vertexs_data {
            pipeline.add_data(verts.clone());
            pipeline.draw(&uniforms);
        }

        // ----- 显示 -----
        window
            .update_with_buffer(pipeline.display_buffer(), width, height)
            .unwrap();

        // ----- FPS 统计 -----
        frame_count += 1;
        let elapsed = fps_timer.elapsed();
        if elapsed.as_secs_f64() >= 1.0 {
            fps = frame_count;
            frame_count = 0;
            fps_timer = Instant::now();

            window.set_title(&format!("Tiny Renderer - {} FPS", fps));

            if DEBUG_FPS {
                println!("[DEBUG] FPS: {}", fps);
            }
        }
    }

    // 退出时保存
    framebuffer.save_to_image("output_render.tga");
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
