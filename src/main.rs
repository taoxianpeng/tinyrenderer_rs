mod african_head;
mod boggie;
mod diablo3_pose;
mod drawline;
mod floor;
mod drawtriangle;
mod model;
mod renderpipeline;
mod tgaimage;
mod framebuffer;

use minifb::{Key, Window, WindowOptions};
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::time::Instant;

use renderpipeline::{RenderPipleline, Uniforms};

use crate::framebuffer::FrameBuffer;
use crate::renderpipeline::{lookat, projection};
use crate::tgaimage::TGAColor;

const DEBUG_FPS: bool = false;

fn main() {
    run();
}

fn run() {
    let head_assets = african_head::load_african_head_assets();
    let boggie_assets = boggie::load_boggie_assets();
    let diablo_assets = diablo3_pose::load_diablo3_pose_assets();
    let floor_assets = floor::load_floor_assets();

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

    // 三个模型沿 x 轴错开摆放（包围盒宽 ≤1.62，间距 2.2 互不重叠）
    let head_offset = Vec3::new(-2.2, 0.0, 0.0);
    let boggie_offset = Vec3::ZERO;
    let diablo_offset = Vec3::new(2.2, 0.0, 0.0);

    // 视野加宽 + 相机拉远 + 远裁剪面放宽，保证三者都在视锥内
    let proj_mat = projection(
        renderpipeline::ProjectionMode::PERSPECTIVE,
        std::f32::consts::FRAC_PI_4,
        Vec2 {
            x: width as f32,
            y: height as f32,
        },
        0.1,
        50.0,
    );

    // 打开 minifb 窗口（不限帧率）
    let mut window = Window::new("Tiny Renderer", width, height, WindowOptions::default())
        .unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0);

    // 相机状态：球面坐标
    let mut yaw = 0.0f32;
    let mut pitch = 0.0f32;
    let radius = 5.5f32;
    let center = Vec3::ZERO;
    let up = Vec3::Y;
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

        // view_dir: 从场景中心指向相机的方向 → 用于 Blinn-Phong 半向量
        let view_dir = (eye - center).normalize();

        // ----- 渲染（复用 pipeline，不复分配）-----
        pipeline.clear_buffer(&bg_color);
        pipeline.begin_frame();

        pipeline.set_vertex_shader(floor::vertex_shader);
        pipeline.set_fragment_shader(floor::fragment_shader);
        // floor.obj 两个三角形的绕序与本工程背面剔除约定相反，画地面时关闭剔除
        pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        let model_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));
        let floor_uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir,
            view_dir,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: Some(&floor_assets.diffuse_texture),
            normal_tex: Some(&floor_assets.normal_texture),
            specular_tex: Some(&floor_assets.spec_texture),
            glossiness_tex: None,
        };
        pipeline.draw(&floor_assets.models[0], &floor_uniforms);
        pipeline.set_cull_mode(renderpipeline::CullMode::BACK);

        // ===== 1. african_head =====
        pipeline.set_vertex_shader(african_head::vertex_shader);
        pipeline.set_fragment_shader(african_head::fragment_shader);
        let model_mat = Mat4::from_translation(head_offset);
        let mut uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir,
            view_dir,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: Some(&head_assets.head_diffuse_texture),
            normal_tex: Some(&head_assets.head_normal_texture),
            specular_tex: Some(&head_assets.head_spec_texture),
            glossiness_tex: None,
        };

        // draw head
        pipeline.draw(&head_assets.models[0], &uniforms);

        // draw inner eye（虹膜，不透明，先画）：眼睛没有独立贴图，关闭法线/高光贴图
        uniforms.normal_tex = Some(&head_assets.eye_inner_normal_texture);
        uniforms.diffuse_tex = Some(&head_assets.eye_inner_diffuse_texture);
        uniforms.specular_tex = None;
        pipeline.draw(&head_assets.models[1], &uniforms);

        // draw outer eye（角膜，diffuse 贴图带 alpha，半透明，后画以混合出虹膜颜色）
        uniforms.normal_tex = Some(&head_assets.eye_outer_normal_texture);
        uniforms.diffuse_tex = Some(&head_assets.eye_outer_diffuse_texture);
        pipeline.draw(&head_assets.models[2], &uniforms);

        // ===== 2. boggie（body / head / eyes 共用同一模型矩阵，部件坐标自带偏移）=====
        pipeline.set_vertex_shader(boggie::vertex_shader);
        pipeline.set_fragment_shader(boggie::fragment_shader);
        let model_mat = Mat4::from_translation(boggie_offset);
        let mut uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir,
            view_dir,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: Some(&boggie_assets.body_diffuse_texture),
            normal_tex: Some(&boggie_assets.body_normal_texture),
            specular_tex: Some(&boggie_assets.body_spec_texture),
            glossiness_tex: None,
        };

        // draw body
        pipeline.draw(&boggie_assets.models[0], &uniforms);

        // draw head
        uniforms.diffuse_tex = Some(&boggie_assets.head_diffuse_texture);
        uniforms.normal_tex = Some(&boggie_assets.head_normal_texture);
        uniforms.specular_tex = Some(&boggie_assets.head_spec_texture);
        pipeline.draw(&boggie_assets.models[1], &uniforms);

        // draw eyes
        uniforms.diffuse_tex = Some(&boggie_assets.eyes_diffuse_texture);
        uniforms.normal_tex = Some(&boggie_assets.eyes_normal_texture);
        uniforms.specular_tex = Some(&boggie_assets.eyes_spec_texture);
        pipeline.draw(&boggie_assets.models[2], &uniforms);

        // ===== 3. diablo3_pose（glow 贴图借 glossiness_tex 通道作为自发光项）=====
        pipeline.set_vertex_shader(diablo3_pose::vertex_shader);
        pipeline.set_fragment_shader(diablo3_pose::fragment_shader);
        let model_mat = Mat4::from_translation(diablo_offset);
        let uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir,
            view_dir,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: Some(&diablo_assets.diffuse_texture),
            normal_tex: Some(&diablo_assets.normal_texture),
            specular_tex: Some(&diablo_assets.spec_texture),
            glossiness_tex: Some(&diablo_assets.glow_texture),
        };

        pipeline.draw(&diablo_assets.models[0], &uniforms);

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
