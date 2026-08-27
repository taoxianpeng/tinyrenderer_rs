use glam::{Mat3, Vec3};

use crate::drawline::TGAImageType::RGB;
use crate::drawline::TGAImage;
use crate::model::load_model;
use crate::renderpipeline::{FragmentInput, Uniforms, Varying, VertexInput, VertexOutput};
use crate::tgaimage::TGAColor;

/// floor 的全部资产：1 个模型 + 3 张贴图
pub struct FloorAssets {
    /// 模型：[0]=地面四边形
    pub models: Vec<Vec<VertexInput>>,
    /// 漫反射纹理 grid.tga (1024²)
    pub diffuse_texture: TGAImage,
    /// 切线空间法线贴图 floor_nm_tangent.tga (512²)
    pub normal_texture: TGAImage,
    /// 高光贴图 floor_spec.tga (1×1 占位图)
    pub spec_texture: TGAImage,
}

/// 一次性加载 floor 的模型与贴图（加载失败直接 panic，符合本工程惯例）
pub fn load_floor_assets() -> FloorAssets {
    // ---- 模型 ----
    let models: Vec<Vec<VertexInput>> = vec![load_model("assert/floor.obj", true)]
        .into_iter()
        .flatten()
        .collect();

    // ---- 贴图 ----
    // 漫反射：grid.tga（网格地面纹理）
    let mut diffuse_texture = TGAImage::new(1024, 1024, RGB);
    diffuse_texture
        .read_tga_file("assert/grid.tga")
        .unwrap();
    diffuse_texture.flip_vertically();

    let mut normal_texture = TGAImage::new(512, 512, RGB);
    normal_texture
        .read_tga_file("assert/floor_nm_tangent.tga")
        .unwrap();
    normal_texture.flip_vertically();

    let mut spec_texture = TGAImage::new(1, 1, RGB);
    spec_texture
        .read_tga_file("assert/floor_spec.tga")
        .unwrap();
    spec_texture.flip_vertically();

    FloorAssets {
        models,
        diffuse_texture,
        normal_texture,
        spec_texture,
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

/// 采样一张贴图的像素；采样器逻辑与 boggie/diablo3_pose 一致
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

    // 采样高光贴图：取 r/g/b 最大值以兼容灰度格式（floor_spec.tga 是 1×1 黑色占位图）
    let spec_mask = match uniforms.specular_tex {
        Some(spec_tex) => match sample_texture(spec_tex, &texcoord) {
            Some(c) => Vec3::splat(c.r.max(c.g).max(c.b)),
            None => Vec3::splat(1.0),
        },
        None => Vec3::splat(1.0),
    };

    let ambient_light_strength = 0.4;
    let ambient = ambient_light_strength * uniforms.ambient_color;

    let diff = f32::max(normal.dot(uniforms.light_dir), 0.0);
    let diffuse = diff * diffuse_color;

    let specular_light_strength = 1.0;
    let halfway_dir = (uniforms.light_dir + uniforms.view_dir).normalize();
    let spec = f32::powi(f32::max(normal.dot(halfway_dir), 0.0), 32);
    let specular = specular_light_strength * spec * uniforms.specular_color * spec_mask;

    let color_t = ambient + diffuse + specular;

    TGAColor::new(color_t.x, color_t.y, color_t.z, diffuse_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec2};

    use crate::framebuffer::FrameBuffer;
    use crate::renderpipeline::{self, lookat, projection, RenderPipleline, Uniforms};

    /// 离屏渲染地面：四边形朝上（法线 +y），相机俯视时放大后的地面应铺满大部分画面
    #[test]
    fn render_floor_covers_view() {
        let verts = load_model("assert/floor.obj", true).unwrap();

        let mut diffuse_texture = TGAImage::new(1024, 1024, RGB);
        diffuse_texture.read_tga_file("assert/grid.tga").unwrap();
        diffuse_texture.flip_vertically();

        let mut normal_texture = TGAImage::new(512, 512, RGB);
        normal_texture
            .read_tga_file("assert/floor_nm_tangent.tga")
            .unwrap();
        normal_texture.flip_vertically();

        let width = 400;
        let height = 400;
        // 相机抬高并后退俯视地面（地面所有顶点都在相机前方，避免被 w≈0 退化剔除），
        // 保证地面法线 (0,1,0) 朝向相机
        let eye = Vec3::new(0.0, 3.0, 10.0);
        let center = Vec3::new(0.0, -1.0, 0.0);
        let up = Vec3::Y;
        let model_mat = Mat4::from_scale(Vec3::new(8.0, 1.0, 8.0));

        let proj_mat = projection(
            renderpipeline::ProjectionMode::PERSPECTIVE,
            std::f32::consts::FRAC_PI_4,
            Vec2::new(width as f32, height as f32),
            0.1,
            50.0,
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
            diffuse_tex: Some(&diffuse_texture),
            normal_tex: Some(&normal_texture),
            specular_tex: None,
            glossiness_tex: None,
        };

        let mut framebuffer = FrameBuffer::new(width, height);
        let mut pipeline = RenderPipleline::new(&mut framebuffer);
        pipeline.set_vertex_shader(vertex_shader);
        pipeline.set_fragment_shader(fragment_shader);
        pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        pipeline.clear_buffer(&TGAColor::new(0.0, 0.0, 0.0, 1.0));
        pipeline.begin_frame();
        pipeline.draw(&verts, &uniforms);

        let mut drawn_pixels = 0usize;
        for y in 0..height {
            for x in 0..width {
                if let Some(c) = framebuffer.get(x, y) {
                    if c.r.max(c.g).max(c.b) > 0.0 {
                        drawn_pixels += 1;
                    }
                }
            }
        }
        // 放大后的地面应覆盖画面的相当比例
        assert!(
            drawn_pixels > (width * height) / 4,
            "地面应铺满画面, drawn={}",
            drawn_pixels
        );
    }
}
