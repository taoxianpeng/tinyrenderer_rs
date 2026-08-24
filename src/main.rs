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

use crate::drawline::TGAImageType::RGB;
use crate::drawline::{TGAImage, WHITE};
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

    
    // body texture
    let mut normal_texture = TGAImage::new(1024, 1024, RGB);
    normal_texture.read_tga_file("assert/african_head/african_head_nm.tga").unwrap();
    normal_texture.flip_vertically();

    let mut diffuse_texture = TGAImage::new(1024, 1024, RGB);
    diffuse_texture.read_tga_file("assert/african_head/african_head_diffuse.tga").unwrap();
    diffuse_texture.flip_vertically();

    // eye texture
    let mut eye_inner_normal_texture = TGAImage::new(256, 256, RGB);
    eye_inner_normal_texture.read_tga_file("assert/african_head/african_head_eye_inner_nm.tga").unwrap();
    eye_inner_normal_texture.flip_vertically();

    let mut eye_inner_diffuse_texture = TGAImage::new( 256, 256, RGB);
    eye_inner_diffuse_texture.read_tga_file("assert/african_head/african_head_eye_inner_diffuse.tga").unwrap();
    eye_inner_diffuse_texture.flip_vertically();

    let mut eye_outer_normal_texture = TGAImage::new(256, 256, RGB);
    eye_outer_normal_texture.read_tga_file("assert/african_head/african_head_eye_outer_nm.tga").unwrap();
    eye_outer_normal_texture.flip_vertically();

    let mut eye_outer_diffuse_texture = TGAImage::new(256, 256, RGB);
    eye_outer_diffuse_texture.read_tga_file("assert/african_head/african_head_eye_outer_diffuse.tga").unwrap();
    eye_outer_diffuse_texture.flip_vertically();

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
    pipeline.set_cull_mode(renderpipeline::CullMode::BACK);
    pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);

    pipeline.set_vertex_shader(renderpipeline::default_vertex_shader);
    pipeline.set_fragment_shader(renderpipeline::default_fragment_shader);

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

        let mut uniforms = Uniforms {
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
            diffuse_tex: Some(&diffuse_texture),
            normal_tex: Some(&normal_texture),
            specular_tex: None,
            glossiness_tex: None,
        };

        // ----- 渲染（复用 pipeline，不复分配）-----
        pipeline.clear_buffer(&bg_color);
        pipeline.begin_frame();

        // for verts in &vertexs_data {
        //     pipeline.draw(verts, &uniforms);
        // }

        // draw head
        pipeline.draw(&vertexs_data[0], &uniforms);

        // draw inner eye（虹膜，不透明，先画）
        uniforms.normal_tex = Some(&eye_inner_normal_texture);
        uniforms.diffuse_tex = Some(&eye_inner_diffuse_texture);
        pipeline.draw(&vertexs_data[1], &uniforms);

        // draw outer eye（角膜，diffuse 贴图带 alpha，半透明，后画以混合出虹膜颜色）
        uniforms.normal_tex = Some(&eye_outer_normal_texture);
        uniforms.diffuse_tex = Some(&eye_outer_diffuse_texture);
        pipeline.draw(&vertexs_data[2], &uniforms);

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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// 离屏渲染真实模型：修复高光后，画面应出现明显亮于漫反射上限的像素
    #[test]
    fn render_african_head_has_specular_highlight() {
        let verts = load_model("assert/african_head/african_head.obj").unwrap();

        let mut normal_texture = TGAImage::new(1024, 1024, RGB);
        normal_texture
            .read_tga_file("assert/african_head/african_head_nm.tga")
            .unwrap();
        normal_texture.flip_vertically();

        let width = 400;
        let height = 400;
        let eye = Vec3::new(0.0, 0.0, 2.5);
        let center = Vec3::ZERO;
        let up = Vec3::Y;
        let model_mat = Mat4::IDENTITY;

        let proj_mat = projection(
            renderpipeline::ProjectionMode::PERSPECTIVE,
            std::f32::consts::FRAC_PI_4,
            Vec2::new(width as f32, height as f32),
            0.1,
            10.0,
        );
        let view_mat = lookat(&eye, &center, &up);
        let uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir: Vec3::new(-1.0, 1.0, 1.0).normalize(),
            view_dir: (eye - center).normalize(),
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: None,
            normal_tex: Some(&normal_texture),
            specular_tex: None,
            glossiness_tex: None,
        };

        let mut framebuffer = FrameBuffer::new(width, height);
        let mut pipeline = RenderPipleline::new(&mut framebuffer);
        pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        pipeline.clear_buffer(&TGAColor::new(0.0, 0.0, 0.0, 1.0));
        pipeline.begin_frame();
        pipeline.draw(&verts, &uniforms);

        // 统计亮度高于漫反射上限（ambient 0.2*0.5 + diffuse 0.7*0.7 = 0.59）的像素数
        // 这些像素只能来自 specular 贡献
        let mut bright_pixels = 0usize;
        let mut drawn_pixels = 0usize;
        for y in 0..height {
            for x in 0..width {
                if let Some(c) = framebuffer.get(x, y) {
                    let lum = c.r.max(c.g).max(c.b);
                    if lum > 0.0 {
                        drawn_pixels += 1;
                    }
                    if lum > 0.62 {
                        bright_pixels += 1;
                    }
                }
            }
        }
        assert!(drawn_pixels > 1000, "模型应被正常渲染, drawn={}", drawn_pixels);
        assert!(
            bright_pixels > 100,
            "高光缺失: 亮度>0.62 的像素仅 {} 个",
            bright_pixels
        );
    }
}
