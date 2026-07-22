mod datatype;
mod drawline;
mod drawtriangle;
mod model;
mod tgaimage;
mod renderpipeline;

use std::{path::Path};
use tgaimage::{TGAColor, TGAImage, TGAImageType};
use glam::{Mat3, Mat4, Vec2, Vec3};

use model::Model;
use renderpipeline::{RenderPipleline, VertexInput, Uniforms};

use crate::{drawline::{BLUE, GREEN, RED, WHITE, BLACK}, renderpipeline::{lookat, projection}};

fn main() {
    // 1. 加载 OBJ 模型
    // let model: Model = Model::new(Path::new("assert/diablo3_pose.obj"));
    let model: Model = Model::new(Path::new("assert/african_head.obj"));
    println!(
        "模型加载成功: {} 顶点, {} 面",
        model.verts().len() - 1,
        model.faces().len() - 1,
    );

    // 2. 构建顶点输入数组
    //    遍历每个三角形面，将每个顶点的位置/法线/纹理坐标组成 VertexInput
    let mut vertices: Vec<VertexInput> = Vec::new();
    for face in model.faces() {
        // 只处理三角形面（通常 obj 导出时已三角化）
        if face.len() == 3 {
            for idx in face {
                let pos   = model.verts()[idx[0] as usize];
                let normal = model.vert_normals()[idx[2] as usize];
                let texcoord = {
                    let vt = model.texture_verts()[idx[1] as usize];
                    Vec2::new(vt.x, vt.y)
                };
                vertices.push(VertexInput {
                    pos,
                    varyings: vec![
                        renderpipeline::Varying::Color(WHITE),
                        renderpipeline::Varying::Vec3(normal),
                        renderpipeline::Varying::Vec2(texcoord),
                    ],
                });
            }
        }
    }
    println!("组装 {} 个顶点输入 ({} 个三角形)", vertices.len(), vertices.len() / 3);

    // 3. 创建帧缓冲
    let width  = 800;
    let height = 800;
    let mut framebuffer = TGAImage::new(width, height, TGAImageType::RGB);
    framebuffer.set_background_color(&TGAColor {
        r: 30.0 / 255.0,
        g: 30.0 / 255.0,
        b: 30.0 / 255.0,
        a: 1.0,
    });

    // 4. 设置相机 / 投影变换
    let model_mat = Mat4::IDENTITY;

    // 把模型摆正：绕 X 轴旋转 -90°，使 OBJ 的 Y-up 转为 Z-up 的世界坐标系
    //（实际由 model matrix 控制，这里保持单位阵 + 调整 camera 位置即可）
    let eye    = Vec3::new(1.0, 0.0, 2.5);
    let center = Vec3::ZERO;
    let up     = Vec3::Y;
    let view_mat = lookat(&eye, &center, &up);

    // let proj_mat = Mat4::perspective_rh_gl(
    let proj_mat = projection(
        renderpipeline::ProjectionMode::PERSPECTIVE,
        std::f32::consts::FRAC_PI_4, // 45° FOV
        // width as f32 / height as f32, // 宽高比
        Vec2 { x: width as f32, y: height as f32},
        0.1,                          // near
        10.0,                         // far
    );

    let model_view      = view_mat * model_mat;
    let model_view_proj = proj_mat * model_view;
    let normal_matrix   = Mat3::from_mat4(model_mat.inverse().transpose());

    let uniforms = Uniforms {
        model: model_mat,
        view: view_mat,
        projection: proj_mat,
        model_view,
        model_view_proj,
        normal_matrix,
        light_dir:      Vec3::new(-1.0, 1.0, 1.0).normalize(),
        view_dir:       (eye - center),
        ambient_color:  Vec3::new(0.5, 0.5, 0.5), // 环境光颜色
        diffuse_color:  Vec3::new(0.7, 0.7, 0.7),
        specular_color: Vec3::new(0.3, 0.3, 0.3),
        diffuse_tex:    None,
        normal_tex:     None,
        specular_tex:   None,
        glossiness_tex: None,
    };

    // 5. 运行渲染管线
    let mut pipeline = RenderPipleline::new(&mut framebuffer);
    pipeline.add_data(&vertices);
    pipeline.set_flat_normal(false);
    pipeline.set_uniforms(&uniforms);
    pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
    pipeline.draw();

    // 6. 输出渲染图
    framebuffer.write_tga_file("output_render.tga", false, true).unwrap();
    println!("输出完成: output_render.tga ({}x{})", width, height);
}
