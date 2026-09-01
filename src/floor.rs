use glam::{Mat3, Vec3, Vec4};

use crate::drawline::TGAImageType::RGB;
use crate::drawline::TGAImage;
use crate::model::load_model;
use crate::renderpipeline::{
    FragmentInput, Uniforms, Varying, VaryingIndex, VertexInput, VertexOutput, in_shadow,
};
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
    let mut diffuse_texture = TGAImage::new(600, 600, RGB);
    diffuse_texture
        .read_tga_file("assert/floor_diffuse.tga")
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
    let local_normal = match input.varyings[VaryingIndex::Normal] {
        Varying::Vec3(n) => n,
        _ => unreachable!("second varying must be normal"),
    };
    // 变换到世界空间
    let world_normal = (uniforms.normal_matrix * local_normal).normalize();

    // 获取局部切线 T（load_model 在加载时按面计算并写入 varyings[3]）
    let local_tangent = match input.varyings[VaryingIndex::Tangent] {
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
    varyings[VaryingIndex::Normal] = Varying::Vec3(world_normal);
    varyings[VaryingIndex::Tangent] = Varying::Vec3(t); // 世界空间 T
    // 槽位已由 load_model 按 VaryingIndex::COUNT 建满，一律按下标赋值（不再 push）
    varyings[VaryingIndex::Bitangent] = Varying::Vec3(b); // varyings[4]: 世界空间 B

    // 阴影映射：世界坐标 → 光源裁剪空间。
    // 必须用 uniforms.model（世界空间）；uniforms.model_view 是 view·model，
    // 处在**当前 pass 的视图空间**，再乘光源 light_view_proj 等于套了两层视图变换。
    // 关闭阴影时也写入占位 Vec4，保证该槽位的类型与下标恒定
    let light_clip = match uniforms.light_view_proj {
        Some(proj) => proj * (uniforms.model * input.pos.extend(1.0)),
        None => Vec4::ZERO,
    };
    varyings[VaryingIndex::LightViewPos] = Varying::Vec4(light_clip); // varyings[5]

    VertexOutput { pos, varyings: Some(varyings) }
}

/// 采样一张贴图的像素；采样器逻辑与 boggie/diablo3_pose 一致
fn sample_texture(tex: &TGAImage, texcoord: &glam::Vec2) -> Option<TGAColor> {
    let x = ((texcoord.x.clamp(0.0, 1.0) * tex.width() as f32) as usize).min(tex.width() - 1);
    let y = ((texcoord.y.clamp(0.0, 1.0) * tex.height() as f32) as usize).min(tex.height() - 1);
    tex.get(x, y)
}

pub fn fragment_shader(uniforms: &Uniforms, frag: &FragmentInput, _depth: &Vec<f32>) -> TGAColor {
    // 插值得到的顶点法线（无 normal map 时的光照法线）
    let vertex_normal = match frag.varyings[VaryingIndex::Normal] {
        Varying::Vec3(normal) => normal.normalize(),
        _ => unreachable!("second varying must be normal"),
    };

    let texcoord = match frag.varyings[VaryingIndex::TexCoord] {
        Varying::Vec2(texcoord) => texcoord,
        _ => unreachable!("third varying must be texcoord"),
    };

    // 插值得到的世界空间 T、B（顶点着色器中已做 Gram-Schmidt 正交化）
    let t = match frag.varyings[VaryingIndex::Tangent] {
        Varying::Vec3(t) => t.normalize(),
        _ => unreachable!("fourth varying must be tangent"),
    };
    let b = match frag.varyings[VaryingIndex::Bitangent] {
        Varying::Vec3(b) => b.normalize(),
        _ => unreachable!("fifth varying must be bitangent"),
    };

    // 光源裁剪坐标：槽位恒定存在（关阴影时是 Vec4::ZERO 占位，in_shadow 内部自会判否）
    let light_clip = match frag.varyings[VaryingIndex::LightViewPos] {
        Varying::Vec4(pos) => pos,
        _ => unreachable!("varyings[5] must be the light-space clip position"),
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

    // 阴影判定交给管线共享的 in_shadow：内部完成光源空间投影、视锥越界与哨兵判断、
    // 斜率缩放 bias，并与 depth_test 存的同一种量（光源裁剪空间 z）直接比较
    let ambient_light_strength = 0.4;
    let ambient = ambient_light_strength * uniforms.ambient_color;

    // N·L 先算：既是漫反射强度，也是阴影 bias 的斜率因子（掠射面需要更大容差）
    let diff = f32::max(normal.dot(uniforms.light_dir), 0.0);
    let shadowed = in_shadow(uniforms, light_clip, diff);
    let diffuse = if shadowed { Vec3::ZERO } else { diff * diffuse_color };

    let specular_light_strength = 1.0;
    let halfway_dir = (uniforms.light_dir + uniforms.view_dir).normalize();
    let spec = f32::powi(f32::max(normal.dot(halfway_dir), 0.0), 32);
    // 阴影里没有直射光，也就不该有直射高光
    let specular = if shadowed {
        Vec3::ZERO
    } else {
        specular_light_strength * spec * uniforms.specular_color * spec_mask
    };

    let color_t = ambient + diffuse + specular;

    TGAColor::new(color_t.x, color_t.y, color_t.z, diffuse_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec2};

    use crate::framebuffer::FrameBuffer;
    use crate::renderpipeline::{
        self, ShadowMap, lookat, projection, RenderPipleline, Uniforms,
    };

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
            light_view_proj: None,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: Some(&diffuse_texture),
            normal_tex: Some(&normal_texture),
            specular_tex: None,
            glossiness_tex: None,
            depth_tex_raw: None,
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

    /// 回归测试：光源正交深度 pass 必须真正写入 shadow map
    /// （曾因为 projection(ORTHO) 把 z 平移项放错行、污染 w，导致场景坍缩成亚像素、
    /// 深度缓冲全为 f32::MAX，D 键查看深度图时表现为全黑）
    #[test]
    fn light_ortho_depth_pass_fills_shadow_map() {
        let verts = load_model("assert/floor.obj", true).unwrap();

        let width = 800;
        let height = 800;
        let light_dir = Vec3::new(-5.0, 5.0, 5.0);
        let view_mat = lookat(&light_dir, &Vec3::ZERO, &Vec3::Y);
        // 与 main.rs 一致：view_size 是世界单位（10×10 罩住 ±2.5 缩放的地面），
        // near/far 夹住场景沿光轴的深度（眼距原点 8.66 ± 场景半径 3.6）
        let proj_mat = projection(
            renderpipeline::ProjectionMode::ORTHO,
            0.0,
            Vec2::new(10.0, 10.0),
            4.0,
            16.0,
        );
        let model_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));

        let uniforms = Uniforms {
            model: model_mat,
            view: view_mat,
            projection: proj_mat,
            model_view: view_mat * model_mat,
            model_view_proj: proj_mat * view_mat * model_mat,
            normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
            light_dir: light_dir.normalize(),
            view_dir: light_dir,
            light_view_proj: None,
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: None,
            normal_tex: None,
            specular_tex: None,
            glossiness_tex: None,
            depth_tex_raw: None,
        };

        let mut framebuffer = FrameBuffer::new(width, height);
        let mut pipeline = RenderPipleline::new(&mut framebuffer);
        pipeline.set_vertex_shader(crate::renderpipeline::default_vertex_shader);
        pipeline.set_fragment_shader(crate::renderpipeline::default_fragment_shader);
        pipeline.set_only_depth_output(true);
        pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        pipeline.begin_frame();
        pipeline.draw(&verts, &uniforms);

        let written = pipeline
            .get_depth_buffer()
            .iter()
            .filter(|d| **d < f32::MAX)
            .count();
        // 10×10 正交窗口里，倾 45° 的 5×5 地面约占 15% 面积，取稳健下限
        assert!(
            written > width * height / 20,
            "光源正交 pass 应把地面写入 shadow map, 实际写入 {}/{} 像素",
            written,
            width * height
        );

        // 深度值必须落在 [-1, 1] 的 NDC 区间内（正交下 clip z == ndc z），
        // 否则说明近/远平面没有夹住场景
        let (min_d, max_d) = pipeline
            .get_depth_buffer()
            .iter()
            .filter(|d| **d < f32::MAX)
            .fold((f32::MAX, f32::MIN), |acc, &d| {
                (acc.0.min(d), acc.1.max(d))
            });
        assert!(
            min_d > -1.0 && max_d < 1.0,
            "场景深度应落在光视锥内, got [{}, {}]",
            min_d,
            max_d
        );
    }

    /// 回归测试：启用阴影映射（view_proj 与 depth_tex_raw 均存在）时，
    /// floor 的顶点+片元链路必须能跑通。
    /// 曾因 VaryingIndex::Bitangent 跳号为 5、而 varyings 实际只有 4 项而数组越界 panic
    /// 功能测试：shadow map 的深度比较必须真正生效。
    /// 用两种极端填充值伪造光源深度图——全为"远"（+0.95，不遮挡）与全为"近"
    /// （-0.95，全遮挡）——地面平均亮度必须显著下降。
    /// 曾出现的地牢：拿 light_clip.w 当深度比较（正交下 w≡1 → 全图误判）
    #[test]
    fn floor_shadow_darkens_floor() {
        let verts = load_model("assert/floor.obj", true).unwrap();

        let width = 400;
        let height = 400;
        let up = Vec3::Y;

        let light_dir = Vec3::new(-5.0, 5.0, 5.0);
        let light_view = lookat(&light_dir, &Vec3::ZERO, &up);
        let light_proj = projection(
            renderpipeline::ProjectionMode::ORTHO,
            0.0,
            Vec2::new(10.0, 10.0),
            4.0,
            16.0,
        );

        let eye = Vec3::new(0.0, 3.0, 10.0);
        let center = Vec3::ZERO;
        let view_mat = lookat(&eye, &center, &up);
        let proj_mat = projection(
            renderpipeline::ProjectionMode::PERSPECTIVE,
            std::f32::consts::FRAC_PI_4,
            Vec2::new(width as f32, height as f32),
            0.1,
            50.0,
        );
        let model_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));

        // 用给定深度值填满 shadow map，渲染一次，返回地面的平均亮度
        let render_with_map = |map_fill: f32| -> (f32, usize) {
            let uniforms = Uniforms {
                model: model_mat,
                view: view_mat,
                projection: proj_mat,
                model_view: view_mat * model_mat,
                model_view_proj: proj_mat * view_mat * model_mat,
                normal_matrix: Mat3::from_mat4(model_mat.inverse().transpose()),
                light_dir: light_dir.normalize(),
                view_dir: (eye - center).normalize(),
                // 相机 pass 里携带的是**光源**矩阵与光源深度图
                light_view_proj: Some(light_proj * light_view),
                ambient_color: Vec3::new(0.5, 0.5, 0.5),
                diffuse_color: Vec3::new(0.7, 0.7, 0.7),
                specular_color: Vec3::new(0.3, 0.3, 0.3),
                diffuse_tex: None,
                normal_tex: None,
                specular_tex: None,
                glossiness_tex: None,
                depth_tex_raw: Some(ShadowMap {
                    data: vec![map_fill; width * height],
                    width,
                    height,
                }),
            };

            let mut framebuffer = FrameBuffer::new(width, height);
            let mut pipeline = RenderPipleline::new(&mut framebuffer);
            pipeline.set_vertex_shader(vertex_shader);
            pipeline.set_fragment_shader(fragment_shader);
            pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
            pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
            pipeline.begin_frame();
            pipeline.draw(&verts, &uniforms);

            let mut sum = 0.0f32;
            let mut n = 0usize;
            for y in 0..height {
                for x in 0..width {
                    let c = framebuffer.get(x, y).unwrap();
                    let b = c.r.max(c.g).max(c.b);
                    if b > 0.0 {
                        sum += b;
                        n += 1;
                    }
                }
            }
            (sum / n as f32, n)
        };

        let (lit, n_lit) = render_with_map(0.95);
        let (shadowed, n_shad) = render_with_map(-0.95);

        assert!(n_lit > 0, "地面应被绘制出来");
        assert_eq!(n_lit, n_shad, "两次渲染的地面覆盖范围应一致");
        assert!(
            shadowed < lit * 0.6,
            "被遮挡时地面应明显变暗, lit={lit:.3}, shadowed={shadowed:.3}"
        );
        // 被遮挡后应只剩环境光：0.4 * 0.5 ≈ 0.2
        assert!(
            (shadowed - 0.2).abs() < 0.05,
            "阴影里应只剩 ambient(≈0.2), got {shadowed:.3}"
        );
    }

    /// 调试用：复刻 main.rs 的两遍渲染，离屏存一张 output_shadow_check.tga 供肉眼检查
    /// 阴影位置与 bias 调参。默认跳过，避免每次 cargo test 都写文件：
    /// `DUMP_SHADOW=1 cargo test debug_dump_shadow_scene`
    #[test]
    fn debug_dump_shadow_scene() {
        if std::env::var("DUMP_SHADOW").is_err() {
            return;
        }
        let w = 500;
        let h = 500;
        let up = Vec3::Y;
        let light_dir = Vec3::new(-5.0, 5.0, 5.0);
        let light_view = lookat(&light_dir, &Vec3::ZERO, &up);
        let light_proj = projection(
            renderpipeline::ProjectionMode::ORTHO,
            0.0,
            Vec2::new(8.0, 8.0),
            4.0,
            16.0,
        );

        let floor_verts = load_model("assert/floor.obj", true).unwrap();
        let head_verts = load_model("assert/african_head/african_head.obj", false).unwrap();
        let floor_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));
        let head_mat = Mat4::from_translation(Vec3::new(-1.5, 0.0, 0.0));

        let mut u = Uniforms {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            model_view: Mat4::IDENTITY,
            model_view_proj: Mat4::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
            light_dir: light_dir.normalize(),
            view_dir: Vec3::Z,
            light_view_proj: Some(light_proj * light_view),
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: None,
            normal_tex: None,
            specular_tex: None,
            glossiness_tex: None,
            depth_tex_raw: None,
        };

        // ---- Pass 1: 光源深度 ----
        let mut lfb = FrameBuffer::new(w, h);
        {
            let mut lp = RenderPipleline::new(&mut lfb);
            lp.set_only_depth_output(true);
            lp.set_vertex_shader(crate::renderpipeline::default_vertex_shader);
            lp.set_fragment_shader(crate::renderpipeline::default_fragment_shader);
            lp.set_draw_mode(renderpipeline::PolygonMode::FILL);
            lp.begin_frame();
            u.view = light_view;
            u.projection = light_proj;
            lp.set_cull_mode(renderpipeline::CullMode::NULL);
            u.model = floor_mat;
            u.model_view = light_view * floor_mat;
            u.model_view_proj = light_proj * light_view * floor_mat;
            u.normal_matrix = Mat3::from_mat4(floor_mat.inverse().transpose());
            lp.draw(&floor_verts, &u);
            lp.set_cull_mode(renderpipeline::CullMode::BACK);
            u.model = head_mat;
            u.model_view = light_view * head_mat;
            u.model_view_proj = light_proj * light_view * head_mat;
            u.normal_matrix = Mat3::from_mat4(head_mat.inverse().transpose());
            lp.draw(&head_verts, &u);
            u.depth_tex_raw = Some(ShadowMap {
                data: lp.get_depth_buffer().clone(),
                width: w,
                height: h,
            });
        }

        // ---- Pass 2: 相机 ----
        let eye = Vec3::new(0.0, 3.0, 10.0);
        let view_mat = lookat(&eye, &Vec3::ZERO, &up);
        let proj_mat = projection(
            renderpipeline::ProjectionMode::PERSPECTIVE,
            std::f32::consts::FRAC_PI_4,
            Vec2::new(w as f32, h as f32),
            0.1,
            50.0,
        );
        let mut diff_tex = TGAImage::new(600, 600, RGB);
        diff_tex.read_tga_file("assert/floor_diffuse.tga").unwrap();
        diff_tex.flip_vertically();

        let mut fb = FrameBuffer::new(w, h);
        {
            let mut p = RenderPipleline::new(&mut fb);
            p.set_only_depth_output(false);
            p.set_vertex_shader(vertex_shader);
            p.set_fragment_shader(fragment_shader);
            p.set_draw_mode(renderpipeline::PolygonMode::FILL);
            p.set_cull_mode(renderpipeline::CullMode::NULL);
            p.clear_buffer(&TGAColor::new(0.12, 0.12, 0.12, 1.0));
            p.begin_frame();
            u.view = view_mat;
            u.projection = proj_mat;
            u.view_dir = (eye - Vec3::ZERO).normalize();
            u.model = floor_mat;
            u.model_view = view_mat * floor_mat;
            u.model_view_proj = proj_mat * view_mat * floor_mat;
            u.normal_matrix = Mat3::from_mat4(floor_mat.inverse().transpose());
            u.diffuse_tex = Some(&diff_tex);
            p.draw(&floor_verts, &u);

            // head 用管线内置 phong 着色器（不测它的阴影，只为画面参照）
            p.set_vertex_shader(crate::renderpipeline::phong_vertex_shader);
            p.set_fragment_shader(crate::renderpipeline::phong_fragment_shader);
            p.set_cull_mode(renderpipeline::CullMode::BACK);
            u.diffuse_tex = None;
            u.model = head_mat;
            u.model_view = view_mat * head_mat;
            u.model_view_proj = proj_mat * view_mat * head_mat;
            u.normal_matrix = Mat3::from_mat4(head_mat.inverse().transpose());
            p.draw(&head_verts, &u);
        }
        fb.save_to_image("output_shadow_check.tga");
    }

    /// 端到端验收：光源 pass 画地面 + head，相机 pass 只画地面，
    /// head 必须在地面上投出一块**局部的**阴影。
    /// 这个断言能抓住两类错误：拿 w 当深度比较（正交下 w≡1 → 阴影铺满全屏）、
    /// 相机 pass 里矩阵被覆盖（→ 阴影完全不出现或位置全错）。
    /// shadow map 尺寸 512² 故意不同于屏幕 400²，顺带验证查表用贴图自身尺寸而非写死分辨率
    #[test]
    fn head_casts_localized_shadow_on_floor() {
        let floor_verts = load_model("assert/floor.obj", true).unwrap();
        let head_verts = load_model("assert/african_head/african_head.obj", false).unwrap();

        let width = 400;
        let height = 400;
        let map_w = 512;
        let map_h = 512;
        let up = Vec3::Y;

        let light_dir = Vec3::new(-5.0, 5.0, 5.0);
        let light_view = lookat(&light_dir, &Vec3::ZERO, &up);
        let light_proj = projection(
            renderpipeline::ProjectionMode::ORTHO,
            0.0,
            Vec2::new(8.0, 8.0),
            4.0,
            16.0,
        );
        let light_view_proj = light_proj * light_view;

        // ---------- Pass 1: 光源深度 pass，画地面 + head（与 main.rs 一致）----------
        // 地面自身也写进 shadow map —— 这正是 self-shadowing 场景：
        // 比较用错分量或 bias 不足时，整块地面会被自己判成阴影
        let floor_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));
        let head_mat = Mat4::from_translation(Vec3::new(-1.5, 0.0, 0.0));
        let mut light_fb = FrameBuffer::new(map_w, map_h);
        let mut light_pipeline = RenderPipleline::new(&mut light_fb);
        light_pipeline.set_only_depth_output(true);
        light_pipeline.set_vertex_shader(crate::renderpipeline::default_vertex_shader);
        light_pipeline.set_fragment_shader(crate::renderpipeline::default_fragment_shader);
        light_pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        light_pipeline.begin_frame();

        let mut lu = Uniforms {
            model: floor_mat,
            view: light_view,
            projection: light_proj,
            model_view: light_view * floor_mat,
            model_view_proj: light_proj * light_view * floor_mat,
            normal_matrix: Mat3::from_mat4(floor_mat.inverse().transpose()),
            light_dir: light_dir.normalize(),
            view_dir: light_dir,
            light_view_proj: Some(light_view_proj),
            ambient_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_tex: None,
            normal_tex: None,
            specular_tex: None,
            glossiness_tex: None,
            depth_tex_raw: None,
        };
        // 地面绕序与背面剔除约定相反 → 光 pass 里同样要关剔除
        light_pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        light_pipeline.draw(&floor_verts, &lu);
        light_pipeline.set_cull_mode(renderpipeline::CullMode::BACK);
        // 切到 head 的模型矩阵
        lu.model = head_mat;
        lu.model_view = light_view * head_mat;
        lu.model_view_proj = light_proj * light_view * head_mat;
        lu.normal_matrix = Mat3::from_mat4(head_mat.inverse().transpose());
        light_pipeline.draw(&head_verts, &lu);
        let shadow_map = ShadowMap {
            data: light_pipeline.get_depth_buffer().clone(),
            width: map_w,
            height: map_h,
        };
        // 前置条件：shadow map 必须有内容，否则下面的断言没有意义
        let map_filled = shadow_map.data.iter().filter(|d| **d < f32::MAX).count();
        assert!(
            map_filled > 1000,
            "地面与 head 应被投影进 shadow map, 实际写入 {map_filled} 纹素"
        );

        // ---------- Pass 2: 相机 pass，只画地面 ----------
        let eye = Vec3::new(0.0, 3.0, 10.0);
        let center = Vec3::ZERO;
        let view_mat = lookat(&eye, &center, &up);
        let proj_mat = projection(
            renderpipeline::ProjectionMode::PERSPECTIVE,
            std::f32::consts::FRAC_PI_4,
            Vec2::new(width as f32, height as f32),
            0.1,
            50.0,
        );
        let floor_mat = Mat4::from_scale(Vec3::new(2.5, 1.0, 2.5));

        lu.model = floor_mat;
        lu.view = view_mat;
        lu.projection = proj_mat;
        lu.model_view = view_mat * floor_mat;
        lu.model_view_proj = proj_mat * view_mat * floor_mat;
        lu.normal_matrix = Mat3::from_mat4(floor_mat.inverse().transpose());
        lu.view_dir = (eye - center).normalize();
        lu.depth_tex_raw = Some(shadow_map); // 光源矩阵 lu.light_view_proj 保持不变

        let mut framebuffer = FrameBuffer::new(width, height);
        let mut pipeline = RenderPipleline::new(&mut framebuffer);
        pipeline.set_vertex_shader(vertex_shader);
        pipeline.set_fragment_shader(fragment_shader);
        pipeline.set_draw_mode(renderpipeline::PolygonMode::FILL);
        pipeline.set_cull_mode(renderpipeline::CullMode::NULL);
        pipeline.begin_frame();
        pipeline.draw(&floor_verts, &lu);

        // ---------- 统计：地面里被遮住的像素应是一小块，而不是全部或全无 ----------
        let mut lit = 0usize;
        let mut dark = 0usize;
        for y in 0..height {
            for x in 0..width {
                let c = framebuffer.get(x, y).unwrap();
                let b = c.r.max(c.g).max(c.b);
                if b <= 0.0 {
                    continue; // 背景，不在地面上
                }
                if b < 0.35 {
                    dark += 1; // 只剩 ambient(≈0.2) → 被遮
                } else {
                    lit += 1; // ambient + diffuse(≈0.6) → 受光
                }
            }
        }
        let floor_px = lit + dark;
        assert!(floor_px > width * height / 8, "地面应铺满画面");
        assert!(dark > 100, "head 应在地面投下可见阴影, dark={dark}");
        assert!(
            dark * 3 < floor_px,
            "阴影应是局部的一块而不是铺满地面, dark={dark}/{floor_px}"
        );
    }
}
