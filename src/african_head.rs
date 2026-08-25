//! african_head 模型强相关代码：资产加载（模型 + 贴图）与离屏渲染测试。

use glam::{Mat3, Vec3};

use crate::drawline::TGAImageType::RGB;
use crate::drawline::TGAImage;
use crate::model::load_model;
use crate::renderpipeline::{FragmentInput, Uniforms, Varying, VertexInput, VertexOutput};
use crate::tgaimage::TGAColor;

/// african_head 的全部资产：3 个模型 + 6 张贴图
pub struct AfricanHeadAssets {
    /// 模型：[0]=头部, [1]=内眼(虹膜), [2]=外眼(角膜)
    pub models: Vec<Vec<VertexInput>>,
    /// 头部法线贴图 (1024²)
    pub head_normal_texture: TGAImage,
    /// 头部漫反射贴图 (1024²)
    pub head_diffuse_texture: TGAImage,
    /// 内眼法线贴图 (256²)
    pub eye_inner_normal_texture: TGAImage,
    /// 内眼漫反射贴图 (256²)
    pub eye_inner_diffuse_texture: TGAImage,
    /// 外眼法线贴图 (256²)
    pub eye_outer_normal_texture: TGAImage,
    /// 外眼漫反射贴图 (256²，带 alpha)
    pub eye_outer_diffuse_texture: TGAImage,
}

/// 一次性加载 african_head 的全部模型与贴图（加载失败直接 panic，符合本工程惯例）
pub fn load_african_head_assets() -> AfricanHeadAssets {
    // ---- 模型 ----
    let models: Vec<Vec<VertexInput>> = vec![
        load_model("assert/african_head/african_head.obj", true),
        load_model("assert/african_head/african_head_eye_inner.obj", true),
        load_model("assert/african_head/african_head_eye_outer.obj", true),
    ]
    .into_iter()
    .flatten()
    .collect();

    // ---- 头部贴图 ----
    let mut head_normal_texture = TGAImage::new(1024, 1024, RGB);
    head_normal_texture
        .read_tga_file("assert/african_head/african_head_nm_tangent.tga")
        .unwrap();
    head_normal_texture.flip_vertically();

    let mut head_diffuse_texture = TGAImage::new(1024, 1024, RGB);
    head_diffuse_texture
        .read_tga_file("assert/african_head/african_head_diffuse.tga")
        .unwrap();
    head_diffuse_texture.flip_vertically();

    // ---- 眼睛贴图 ----
    let mut eye_inner_normal_texture = TGAImage::new(256, 256, RGB);
    eye_inner_normal_texture
        .read_tga_file("assert/african_head/african_head_eye_inner_nm_tangent.tga")
        .unwrap();
    eye_inner_normal_texture.flip_vertically();

    let mut eye_inner_diffuse_texture = TGAImage::new(256, 256, RGB);
    eye_inner_diffuse_texture
        .read_tga_file("assert/african_head/african_head_eye_inner_diffuse.tga")
        .unwrap();
    eye_inner_diffuse_texture.flip_vertically();

    let mut eye_outer_normal_texture = TGAImage::new(256, 256, RGB);
    eye_outer_normal_texture
        .read_tga_file("assert/african_head/african_head_eye_outer_nm_tangent.tga")
        .unwrap();
    eye_outer_normal_texture.flip_vertically();

    let mut eye_outer_diffuse_texture = TGAImage::new(256, 256, RGB);
    eye_outer_diffuse_texture
        .read_tga_file("assert/african_head/african_head_eye_outer_diffuse.tga")
        .unwrap();
    eye_outer_diffuse_texture.flip_vertically();

    AfricanHeadAssets {
        models,
        head_normal_texture,
        head_diffuse_texture,
        eye_inner_normal_texture,
        eye_inner_diffuse_texture,
        eye_outer_normal_texture,
        eye_outer_diffuse_texture,
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
    // 右手系约定（与 *_nm_tangent.tga 烘焙手性一致，已实测验证）：B = cross(N, T)
    let b = world_normal.cross(t);

    let mut varyings = input.varyings.clone();
    varyings[1] = Varying::Vec3(world_normal);
    varyings[3] = Varying::Vec3(t); // 世界空间 T
    varyings.push(Varying::Vec3(b)); // varyings[4]: 世界空间 B

    VertexOutput { pos, varyings }
}

pub fn fragment_shader(uniforms: &Uniforms, frag: &FragmentInput) -> TGAColor {
    // let color = match frag.varyings[0] {
    //     Varying::Color(color) => color,
    //     _ => unreachable!("first varying must be color"),
    // };
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
        Some(normal_image) => {
            let x = ((texcoord.x.clamp(0.0, 1.0) * normal_image.width() as f32) as usize)
                .min(normal_image.width() - 1);
            let y = ((texcoord.y.clamp(0.0, 1.0) * normal_image.height() as f32) as usize)
                .min(normal_image.height() - 1);

            if let Some(normal_color) = normal_image.get(x, y) {
                let n_tangent = (normal_color.to_RGB() * 2.0 - Vec3::splat(1.0)).normalize();
                // 切线空间 → 世界空间
                (tbn * n_tangent).normalize()
            } else {
                vertex_normal
            }
        }
        None => vertex_normal,
    };
    // 采样漫反射贴图（同时取出 alpha，供透明混合使用）
    let (diffuse_color, diffuse_alpha) = match uniforms.diffuse_tex {
        Some(diffuse_tex) => {
            let x = ((texcoord.x.clamp(0.0, 1.0) * diffuse_tex.width() as f32) as usize)
                .min(diffuse_tex.width() - 1);
            let y = ((texcoord.y.clamp(0.0, 1.0) * diffuse_tex.height() as f32) as usize)
                .min(diffuse_tex.height() - 1);
            if let Some(diffuse_color) = diffuse_tex.get(x, y) {
                (diffuse_color.to_RGB(), diffuse_color.a)
            } else {
                (uniforms.diffuse_color, 1.0)
            }
        }
        None => (uniforms.diffuse_color, 1.0),
    };

    let ambient_light_strength = 0.4;
    let ambient = ambient_light_strength * uniforms.ambient_color;

    let diff = f32::max(normal.dot(uniforms.light_dir), 0.0);
    let diffuse = diff * diffuse_color;

    let specular_light_strength = 1.0;
    let halfway_dir = (uniforms.light_dir + uniforms.view_dir).normalize();
    let spec = f32::powi(f32::max(normal.dot(halfway_dir), 0.0), 32);
    let specular = specular_light_strength * spec * uniforms.specular_color;

    let color_t = ambient + diffuse + specular;

    TGAColor::new(color_t.x, color_t.y, color_t.z, diffuse_alpha)

}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Mat4, Vec2, Vec3};

    use crate::framebuffer::FrameBuffer;
    use crate::renderpipeline::{self, lookat, projection, RenderPipleline, Uniforms};
    use crate::tgaimage::TGAColor;

    /// 离屏渲染真实模型：修复高光后，画面应出现明显亮于漫反射上限的像素
    #[test]
    fn render_african_head_has_specular_highlight() {
        let verts = load_model("assert/african_head/african_head.obj", true).unwrap();

        let mut normal_texture = TGAImage::new(1024, 1024, RGB);
        normal_texture
            .read_tga_file("assert/african_head/african_head_nm_tangent.tga")
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
