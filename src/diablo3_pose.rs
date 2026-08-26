//! diablo3_pose 模型强相关代码：资产加载（模型 + 贴图）与离屏渲染测试。
//! 结构仿照 african_head：单模型 + diffuse / 法线 / specular / glow 四张贴图。
//! glow 贴图作为自发光（emissive）项直接叠加到最终颜色上。

use glam::{Mat3, Vec3};

use crate::drawline::TGAImageType::RGB;
use crate::drawline::TGAImage;
use crate::model::load_model;
use crate::renderpipeline::{FragmentInput, Uniforms, Varying, VertexInput, VertexOutput};
use crate::tgaimage::TGAColor;

/// diablo3_pose 的全部资产：1 个模型 + 4 张贴图
pub struct Diablo3PoseAssets {
    /// 模型：[0]=diablo3_pose 整体
    pub models: Vec<Vec<VertexInput>>,
    /// 漫反射贴图 (1024²)
    pub diffuse_texture: TGAImage,
    /// 切线空间法线贴图 (1024²)
    pub normal_texture: TGAImage,
    /// 高光贴图 (1024²)
    pub spec_texture: TGAImage,
    /// 自发光贴图 (1024²)：火焰/符文等发光区域
    pub glow_texture: TGAImage,
}

/// 一次性加载 diablo3_pose 的模型与贴图（加载失败直接 panic，符合本工程惯例）
pub fn load_diablo3_pose_assets() -> Diablo3PoseAssets {
    // ---- 模型 ----
    let models: Vec<Vec<VertexInput>> = vec![load_model(
        "assert/diablo3_pose/diablo3_pose.obj",
        true,
    )]
    .into_iter()
    .flatten()
    .collect();

    // ---- 贴图 ----
    let mut diffuse_texture = TGAImage::new(1024, 1024, RGB);
    diffuse_texture
        .read_tga_file("assert/diablo3_pose/diablo3_pose_diffuse.tga")
        .unwrap();
    diffuse_texture.flip_vertically();

    let mut normal_texture = TGAImage::new(1024, 1024, RGB);
    normal_texture
        .read_tga_file("assert/diablo3_pose/diablo3_pose_nm_tangent.tga")
        .unwrap();
    normal_texture.flip_vertically();

    let mut spec_texture = TGAImage::new(1024, 1024, RGB);
    spec_texture
        .read_tga_file("assert/diablo3_pose/diablo3_pose_spec.tga")
        .unwrap();
    spec_texture.flip_vertically();

    let mut glow_texture = TGAImage::new(1024, 1024, RGB);
    glow_texture
        .read_tga_file("assert/diablo3_pose/diablo3_pose_glow.tga")
        .unwrap();
    glow_texture.flip_vertically();

    Diablo3PoseAssets {
        models,
        diffuse_texture,
        normal_texture,
        spec_texture,
        glow_texture,
    }
}

pub fn vertex_shader(uniforms: &Uniforms, input: &VertexInput) -> VertexOutput {
    let pos = uniforms.model_view_proj * input.pos.extend(1.0);

    // 获取局部法线
    let local_normal = match input.varyings[1] {
        Varying::Vec3(n) => n,
        _ => unreachable!("second varying must be normal"),
    };
    // 变换到世界空间
    let world_normal = (uniforms.normal_matrix * local_normal).normalize();

    // 获取局部切线 T（load_model 在加载时按面计算并写入 varyings[3]）
    let local_tangent = match input.varyings[3] {
        Varying::Vec3(t) => t,
        _ => unreachable!("fourth varying must be tangent"),
    };
    // 变换到世界空间
    let world_tangent = uniforms.normal_matrix * local_tangent;

    // Gram-Schmidt 正交化：去掉 T 中与 N 平行的分量，保证 TBN 三轴两两正交
    let t_ortho = world_tangent - world_normal * world_tangent.dot(world_normal);
    let t = if t_ortho.length_squared() > 1e-6 {
        t_ortho.normalize()
    } else {
        // 退化（T 为 0 或与 N 平行）：任取一个与 N 正交的方向兜底
        let helper = if world_normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        world_normal.cross(helper).normalize()
    };
    // 右手系约定（与 *_nm_tangent.tga 烘焙手性一致）：B = cross(N, T)
    let b = world_normal.cross(t);

    let mut varyings = input.varyings.clone();
    varyings[1] = Varying::Vec3(world_normal);
    varyings[3] = Varying::Vec3(t); // 世界空间 T
    varyings.push(Varying::Vec3(b)); // varyings[4]: 世界空间 B

    VertexOutput { pos, varyings }
}

/// 采样一张贴图的像素；采样器逻辑与 african_head 一致
fn sample_texture(tex: &TGAImage, texcoord: &glam::Vec2) -> Option<TGAColor> {
    let x = ((texcoord.x.clamp(0.0, 1.0) * tex.width() as f32) as usize).min(tex.width() - 1);
    let y = ((texcoord.y.clamp(0.0, 1.0) * tex.height() as f32) as usize).min(tex.height() - 1);
    tex.get(x, y)
}

pub fn fragment_shader(uniforms: &Uniforms, frag: &FragmentInput, _depth: &Vec<f32>) -> TGAColor {
    // 插值得到的顶点法线（无 normal map 时的光照法线）
    let vertex_normal = match frag.varyings[1] {
        Varying::Vec3(normal) => normal.normalize(),
        _ => unreachable!("second varying must be normal"),
    };

    let texcoord = match frag.varyings[2] {
        Varying::Vec2(texcoord) => texcoord,
        _ => unreachable!("third varying must be texcoord"),
    };

    // 插值得到的世界空间 T、B（顶点着色器中已做 Gram-Schmidt 正交化）
    let t = match frag.varyings[3] {
        Varying::Vec3(t) => t.normalize(),
        _ => unreachable!("fourth varying must be tangent"),
    };
    let b = match frag.varyings[4] {
        Varying::Vec3(b) => b.normalize(),
        _ => unreachable!("fifth varying must be bitangent"),
    };
    // TBN 矩阵：列 [T | B | N]，将切线空间方向变换到世界空间
    let tbn = Mat3::from_cols(t, b, vertex_normal);

    // 采样切线空间法线贴图：颜色 [0,1] 解码为方向 [-1,1]
    let normal = match uniforms.normal_tex {
        Some(normal_image) => match sample_texture(normal_image, &texcoord) {
            Some(normal_color) => {
                let n_tangent = (normal_color.to_RGB() * 2.0 - Vec3::splat(1.0)).normalize();
                // 切线空间 → 世界空间
                (tbn * n_tangent).normalize()
            }
            None => vertex_normal,
        },
        None => vertex_normal,
    };

    // 采样漫反射贴图（同时取出 alpha，供透明混合使用）
    let (diffuse_color, diffuse_alpha) = match uniforms.diffuse_tex {
        Some(diffuse_tex) => match sample_texture(diffuse_tex, &texcoord) {
            Some(diffuse_color) => (diffuse_color.to_RGB(), diffuse_color.a),
            None => (uniforms.diffuse_color, 1.0),
        },
        None => (uniforms.diffuse_color, 1.0),
    };

    // 采样高光贴图：灰度值调制 specular 强度（未提供时保持 uniforms.specular_color 原强度）
    let spec_mask = match uniforms.specular_tex {
        Some(spec_tex) => match sample_texture(spec_tex, &texcoord) {
            Some(c) => c.to_RGB(),
            None => Vec3::splat(1.0),
        },
        None => Vec3::splat(1.0),
    };

    // 采样自发光贴图：glossiness_tex 字段在本模块中借用作 glow 贴图通道
    let emissive = match uniforms.glossiness_tex {
        Some(glow_tex) => match sample_texture(glow_tex, &texcoord) {
            Some(c) => c.to_RGB(),
            None => Vec3::ZERO,
        },
        None => Vec3::ZERO,
    };

    let ambient_light_strength = 0.4;
    let ambient = ambient_light_strength * uniforms.ambient_color;

    let diff = f32::max(normal.dot(uniforms.light_dir), 0.0);
    let diffuse = diff * diffuse_color;

    let specular_light_strength = 1.0;
    let halfway_dir = (uniforms.light_dir + uniforms.view_dir).normalize();
    let spec = f32::powi(f32::max(normal.dot(halfway_dir), 0.0), 32);
    let specular = specular_light_strength * spec * uniforms.specular_color * spec_mask;

    let color_t = ambient + diffuse + specular + emissive;

    TGAColor::new(color_t.x, color_t.y, color_t.z, diffuse_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec2};

    use crate::framebuffer::FrameBuffer;
    use crate::renderpipeline::{self, lookat, projection, RenderPipleline, Uniforms};

    /// 离屏渲染 diablo3_pose：画面应出现明显亮于漫反射上限的高光像素
    #[test]
    fn render_diablo3_pose_has_specular_highlight() {
        let verts = load_model("assert/diablo3_pose/diablo3_pose.obj", true).unwrap();

        let mut normal_texture = TGAImage::new(1024, 1024, RGB);
        normal_texture
            .read_tga_file("assert/diablo3_pose/diablo3_pose_nm_tangent.tga")
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
        pipeline.set_vertex_shader(vertex_shader);
        pipeline.set_fragment_shader(fragment_shader);
        pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        pipeline.clear_buffer(&TGAColor::new(0.0, 0.0, 0.0, 1.0));
        pipeline.begin_frame();
        pipeline.draw(&verts, &uniforms);

        // 统计亮度高于漫反射上限（ambient 0.4*0.5 + diffuse 0.7*0.7 = 0.69）的像素数
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
                    if lum > 0.72 {
                        bright_pixels += 1;
                    }
                }
            }
        }
        assert!(drawn_pixels > 1000, "模型应被正常渲染, drawn={}", drawn_pixels);
        assert!(
            bright_pixels > 100,
            "高光缺失: 亮度>0.72 的像素仅 {} 个",
            bright_pixels
        );
    }
}
