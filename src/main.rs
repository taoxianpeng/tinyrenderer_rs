mod african_head;
mod boggie;
mod diablo3_pose;
mod drawline;
mod drawtriangle;
mod floor;
mod framebuffer;
mod model;
mod renderpipeline;
mod tgaimage;

use glam::{Mat3, Mat4, Vec2, Vec3};
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

use renderpipeline::{RenderPipleline, Uniforms};

use crate::framebuffer::FrameBuffer;
use crate::renderpipeline::{lookat, projection};
use crate::tgaimage::TGAColor;

const DEBUG_FPS: bool = false;

fn main() {
    run();
}

/// 复用同一个 uniforms：只更新模型相关字段。
/// view / projection / 光照 / 颜色等由调用方预先写入并保持不变。
fn set_model(uniforms: &mut Uniforms, model_mat: Mat4) {
    uniforms.model = model_mat;
    uniforms.model_view = uniforms.view * model_mat;
    uniforms.model_view_proj = uniforms.projection * uniforms.view * model_mat;
    uniforms.normal_matrix = Mat3::from_mat4(model_mat.inverse().transpose());
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

    let z_near = 0.1f32;
    let z_far = 50.0f32;

    let mut framebuffer = FrameBuffer::new(width, height);
    let mut pipeline = RenderPipleline::new(&mut framebuffer);
    pipeline.set_flat_normal(false);
    pipeline.set_cull_mode(renderpipeline::CullMode::BACK);
    pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);

    // 三个模型沿 x 轴错开摆放（包围盒宽 ≤1.62，间距 2.2 互不重叠）
    let head_offset = Vec3::new(-1.5, 0.0, 0.0);
    let boggie_offset = Vec3::ZERO;
    let diablo_offset = Vec3::new(1.5, 0.0, 0.0);

    // 视野加宽 + 相机拉远 + 远裁剪面放宽，保证三者都在视锥内
    let proj_mat = projection(
        renderpipeline::ProjectionMode::PERSPECTIVE,
        std::f32::consts::FRAC_PI_4,
        Vec2 {
            x: width as f32,
            y: height as f32,
        },
        z_near,
        z_far,
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

    // 整个渲染循环复用同一个 uniforms：相机/光照/颜色每帧更新一次，
    // 模型矩阵与贴图在每次 draw 前更新
    let mut uniforms = Uniforms {
        model: Mat4::IDENTITY,
        view: Mat4::IDENTITY,
        projection: proj_mat,
        model_view: Mat4::IDENTITY,
        model_view_proj: Mat4::IDENTITY,
        normal_matrix: Mat3::IDENTITY,
        light_dir,
        view_dir: Vec3::Z,
        ambient_color: Vec3::new(0.5, 0.5, 0.5),
        diffuse_color: Vec3::new(0.7, 0.7, 0.7),
        specular_color: Vec3::new(0.3, 0.3, 0.3),
        diffuse_tex: None,
        normal_tex: None,
        specular_tex: None,
        glossiness_tex: None,
    };

    let mut fps_timer = Instant::now();
    let mut frame_count = 0u32;
    let mut fps = 0u32;

    // 深度 pass 模式：只渲染深度、跳过着色与颜色写入。
    // 当前由 D 键切换用于实时查看深度图；后续阴影映射作业可复用该状态做光源视角的深度预pass
    let mut only_depth_output = false;
    // 深度图显示用的复用缓冲（0x00RRGGBB 灰度）
    let mut depth_view: Vec<u32> = Vec::with_capacity(width * height);

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
        // D 键松开时切换深度 pass 模式（is_key_released 天然边沿触发，按住只切换一次）
        if window.is_key_released(Key::D) {
            only_depth_output = !only_depth_output;
        }

        pipeline.set_only_depth_output(only_depth_output);

        // ----- 相机位置（球面 → 笛卡尔）-----
        let eye = radius
            * Vec3::new(
                yaw.sin() * pitch.cos(),
                pitch.sin(),
                yaw.cos() * pitch.cos(),
            );

        let view_mat = lookat(&eye, &center, &up);

        // view_dir: 从场景中心指向相机的方向 → 用于 Blinn-Phong 半向量
        let view_dir = (eye - center).normalize();

        // 每帧更新一次相机相关字段（各模型间保持不变）
        uniforms.view = view_mat;
        uniforms.view_dir = view_dir;

        // ----- 帧模式：深度 pass / 正常渲染 -----
        if only_depth_output {
            // 深度 pass：无 varying 顶点着色器；片元着色器换成空实现兜底——
            // 即使管线开关与着色器选择错位，也不会对空 varyings 取下标 panic
            pipeline.set_vertex_shader(renderpipeline::default_vertex_shader);
            pipeline.set_fragment_shader(renderpipeline::default_fragment_shader);
        } else {
            pipeline.set_vertex_shader(floor::vertex_shader);
            pipeline.set_fragment_shader(floor::fragment_shader);
            pipeline.clear_buffer(&bg_color);
        }
        pipeline.begin_frame();

        // floor.obj 两个三角形的绕序与本工程背面剔除约定相反，画地面时关闭剔除
        pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        set_model(&mut uniforms, Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5)));
        uniforms.diffuse_tex = Some(&floor_assets.diffuse_texture);
        uniforms.normal_tex = Some(&floor_assets.normal_texture);
        uniforms.specular_tex = Some(&floor_assets.spec_texture);
        uniforms.glossiness_tex = None;
        pipeline.draw(&floor_assets.models[0], &uniforms);
        pipeline.set_cull_mode(renderpipeline::CullMode::BACK);

        // ===== 1. african_head =====
        if !only_depth_output {
            pipeline.set_vertex_shader(african_head::vertex_shader);
            pipeline.set_fragment_shader(african_head::fragment_shader);
        }

        set_model(&mut uniforms, Mat4::from_translation(head_offset));
        uniforms.diffuse_tex = Some(&head_assets.head_diffuse_texture);
        uniforms.normal_tex = Some(&head_assets.head_normal_texture);
        uniforms.specular_tex = Some(&head_assets.head_spec_texture);
        uniforms.glossiness_tex = None;

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

        // ===== 2. boggie（body / head / eyes ）=====
        if !only_depth_output {
            pipeline.set_vertex_shader(boggie::vertex_shader);
            pipeline.set_fragment_shader(boggie::fragment_shader);
        }
        set_model(&mut uniforms, Mat4::from_translation(boggie_offset));
        uniforms.diffuse_tex = Some(&boggie_assets.body_diffuse_texture);
        uniforms.normal_tex = Some(&boggie_assets.body_normal_texture);
        uniforms.specular_tex = Some(&boggie_assets.body_spec_texture);
        uniforms.glossiness_tex = None;

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
        if !only_depth_output {
            pipeline.set_vertex_shader(diablo3_pose::vertex_shader);
            pipeline.set_fragment_shader(diablo3_pose::fragment_shader);
        }
        set_model(&mut uniforms, Mat4::from_translation(diablo_offset));
        uniforms.diffuse_tex = Some(&diablo_assets.diffuse_texture);
        uniforms.normal_tex = Some(&diablo_assets.normal_texture);
        uniforms.specular_tex = Some(&diablo_assets.spec_texture);
        uniforms.glossiness_tex = Some(&diablo_assets.glow_texture);

        pipeline.draw(&diablo_assets.models[0], &uniforms);

        // ----- 显示 -----
        if only_depth_output {
            // 深度图：按场景实际深度范围 min-max 归一化（固定 [near, far] 线性映射会把
            // 场景灰度压在极窄区间，只剩黑白两色）；背景（f32::MAX 哨兵）置黑
            let mut scene_min = f32::MAX;
            let mut scene_max = f32::MIN;
            for d in pipeline.get_depth_buffer().iter().copied() {
                if d < f32::MAX {
                    scene_min = scene_min.min(d);
                    scene_max = scene_max.max(d);
                }
            }
            let range = scene_max - scene_min;
            depth_view.clear();
            for depth in pipeline.get_depth_buffer() {
                let gray = if *depth >= f32::MAX {
                    0.0f32 // 背景：没有几何写入过
                } else if range > 1e-9 {
                    1.0 - (depth - scene_min) / range // 近 → 白，远 → 黑
                } else {
                    1.0 // 场景深度单一，避免除零
                };
                let v = (gray * 255.0) as u8;
                // 适配深度图显示到屏幕上进行类型转化
                depth_view.push(((v as u32) << 16) | ((v as u32) << 8) | v as u32);
            }
            window
                .update_with_buffer(&depth_view, width, height)
                .unwrap();
        } else {
            window
                .update_with_buffer(pipeline.display_buffer(), width, height)
                .unwrap();
        }

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
    if only_depth_output {
        let mut depth_buffer = FrameBuffer::new(width, height);
        depth_buffer.set_buffer(depth_view);
        depth_buffer.save_to_image("output_depth_render.tga");
    } else {
        framebuffer.save_to_image("output_render.tga");
    }
}
