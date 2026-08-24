use crate::drawline::TGAColor;
use crate::framebuffer::FrameBufferTarget;
use crate::renderpipeline::{ProjectionMode::ORTHO, ProjectionMode::PERSPECTIVE};
use crate::tgaimage::TGAImage;
use glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4, vec4};
use std::cmp::{max, min};
use std::vec;

pub enum PolygonMode {
    FILL,
    LINE,
    Point,
}

pub enum CullMode {
    BACK,
    FRONT,
    NULL,
}

pub fn lookat(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat4 {
    // 先算左向量
    let f = (center - eye).normalize(); // -f -> +z direction 
    let s = f.cross(up.clone()).normalize(); // +x direction
    let u = s.cross(f.clone()).normalize(); // +y direction 

    let x_offset = -s.dot(eye.clone());
    let y_offset = -u.dot(eye.clone());
    let z_offset = f.dot(eye.clone());

    // glam Mat4 是列主序，from_cols 接收 4 个列向量
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(x_offset, y_offset, z_offset, 1.0),
    )
}

pub enum ProjectionMode {
    ORTHO,
    PERSPECTIVE,
}

pub fn projection(
    mode: ProjectionMode,
    fov: f32,
    // aspect_ratio: f32,
    view_size: Vec2, // [width, height]
    z_near: f32,
    z_far: f32,
) -> Mat4 {
    match mode {
        ORTHO => {
            return Mat4::from_cols(
                vec4(2.0 / view_size.x, 0.0, 0.0, 0.0),
                vec4(0.0, 2.0 / view_size.y, 0.0, 0.0),
                vec4(
                    0.0,
                    0.0,
                    2.0 / (z_far - z_near),
                    -(z_far + z_near) / (z_far - z_near),
                ),
                vec4(0.0, 0.0, 0.0, 1.0),
            );
        }
        PERSPECTIVE => {
            let aspect_ratio = view_size.x / view_size.y;
            let tan_fov_div_2 = (fov / 2.0).tan();
            let m_33 = -(z_near + z_far) / (z_far - z_near);
            let m_34 = -2.0 * z_near * z_far / (z_far - z_near);

            Mat4::from_cols(
                vec4(1.0 / (aspect_ratio * tan_fov_div_2), 0.0, 0.0, 0.0),
                vec4(0.0, 1.0 / tan_fov_div_2, 0.0, 0.0),
                vec4(0.0, 0.0, m_33, -1.0),
                vec4(0.0, 0.0, m_34, 0.0),
            )
        }
    }
}

fn max_3(a: i32, b: i32, c: i32) -> i32 {
    max(max(a, b), c)
}

fn min_3(a: i32, b: i32, c: i32) -> i32 {
    min(min(a, b), c)
}

fn is_top_left_edge(v_start: &IVec2, v_end: &IVec2) -> bool {
    // 判断边是否是上边和左边
    let edge = v_end - v_start;

    // 上边界判断
    if edge.y == 0 {
        return edge.x < 0;
    }

    // 左边界判断
    return edge.y < 0;
}

fn is_in_edge(p: &IVec2, v_start: &IVec2, v_end: &IVec2) -> bool {
    return (p.x >= v_start.x && p.x <= v_end.x) && (p.y >= v_start.y && p.y <= v_end.y);
}

pub type VertexShader = fn(&Uniforms<'_>, &VertexInput) -> VertexOutput;
pub type FragmentShader = fn(&Uniforms<'_>, &FragmentInput) -> TGAColor;

pub fn default_vertex_shader(uniforms: &Uniforms, input: &VertexInput) -> VertexOutput {
    let pos = uniforms.model_view_proj * input.pos.extend(1.0);

    // 获取局部法线
    let local_normal = match input.varyings[1] {
        Varying::Vec3(n) => n,
        _ => unreachable!(),
    };
    // 变换到世界空间
    let world_normal = uniforms.normal_matrix * local_normal;

    let mut varyings = input.varyings.clone();
    varyings[1] = Varying::Vec3(world_normal);

    VertexOutput { pos, varyings }
}

pub fn default_fragment_shader(uniforms: &Uniforms, frag: &FragmentInput) -> TGAColor {
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

    // 采样法线贴图：颜色通道 [0,1] 需解码为方向 [-1,1]（object-space 法线贴图）
    let normal = match uniforms.normal_tex {
        Some(normal_image) => {
            let x = ((texcoord.x.clamp(0.0, 1.0) * normal_image.width() as f32) as usize)
                .min(normal_image.width() - 1);
            let y = ((texcoord.y.clamp(0.0, 1.0) * normal_image.height() as f32) as usize)
                .min(normal_image.height() - 1);

            if let Some(normal_color) = normal_image.get(x, y) {
                (normal_color.to_RGB() * 2.0 - Vec3::splat(1.0)).normalize()
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

pub struct RenderPipleline<'a> {
    polygon_mode: PolygonMode,
    flat_normal: bool,
    cull: CullMode,
    framebuffer: &'a mut dyn FrameBufferTarget,
    w: usize,
    h: usize,
    depth_buffer: Vec<f32>,
    color_buffer: Vec<TGAColor>,
    // 可注入的着色器回调（默认使用内置 Blinn-Phong 实现）
    vertex_shader: VertexShader,
    fragment_shader: FragmentShader,
}

impl<'a> RenderPipleline<'a> {
    pub fn new(framebuffer: &'a mut dyn FrameBufferTarget) -> RenderPipleline<'a> {
        let w = framebuffer.width();
        let h = framebuffer.height();
        let total_pixels = w * h;
        RenderPipleline {
            polygon_mode: PolygonMode::LINE,
            flat_normal: false,
            cull: CullMode::NULL,
            framebuffer,
            w,
            h,
            depth_buffer: vec![f32::MAX; total_pixels],
            color_buffer: vec![TGAColor::new(0.0, 0.0, 0.0, 0.0); total_pixels],
            vertex_shader: default_vertex_shader,
            fragment_shader: default_fragment_shader,
        }
    }

    /// 每帧开始时调用：清空顶点数据、重置 depth/color buffer（不复分配）
    pub fn begin_frame(&mut self) {
        self.depth_buffer.fill(f32::MAX);
        self.color_buffer.fill(TGAColor::new(0.0, 0.0, 0.0, 0.0));
    }

    pub fn set_draw_mode(&mut self, mode: PolygonMode) {
        self.polygon_mode = mode;
    }

    pub fn set_flat_normal(&mut self, enable: bool) {
        self.flat_normal = enable;
    }

    pub fn set_cull_mode(&mut self, mode: CullMode) {
        self.cull = mode;
    }

    /// 注入自定义顶点着色器（默认使用内置 Blinn-Phong 顶点着色器）
    pub fn set_vertex_shader(&mut self, shader: VertexShader) {
        self.vertex_shader = shader;
    }

    /// 注入自定义片元着色器（默认使用内置 Blinn-Phong 片元着色器）
    pub fn set_fragment_shader(&mut self, shader: FragmentShader) {
        self.fragment_shader = shader;
    }

    /// 获取帧缓冲的 u32 切片供 minifb 显示
    pub fn display_buffer(&self) -> &[u32] {
        self.framebuffer.raw_buffer()
    }

    /// 清除帧缓冲
    pub fn clear_buffer(&mut self, color: &TGAColor) {
        self.framebuffer.clear(color);
    }

    pub fn draw<'b>(&mut self, vertex_array: &[VertexInput], uniforms: &Uniforms<'b>) {
        let mut primitive_array: Vec<[VertexOutput; 3]> = Vec::new();

        // 每 3 个顶点为一组（一个三角形），执行 vertex shader
        for chunk in vertex_array.chunks(3) {
            if chunk.len() < 3 {
                break; // 不足 3 个顶点，丢弃
            }

            let v0 = (self.vertex_shader)(uniforms, &chunk[0]);
            let v1 = (self.vertex_shader)(uniforms, &chunk[1]);
            let v2 = (self.vertex_shader)(uniforms, &chunk[2]);

            primitive_array.push([v0, v1, v2]);
        }

        // flat normal: 如果开启，将每个三角形三个顶点的法线统一为平均值
        if self.flat_normal {
            for tri in primitive_array.iter_mut() {
                let avg = match (
                    &tri[0].varyings[1],
                    &tri[1].varyings[1],
                    &tri[2].varyings[1],
                ) {
                    (Varying::Vec3(a), Varying::Vec3(b), Varying::Vec3(c)) => {
                        (*a + *b + *c).normalize()
                    }
                    _ => unreachable!("second varying must be normal"),
                };
                let flat = Varying::Vec3(avg);
                tri[0].varyings[1] = flat;
                tri[1].varyings[1] = flat;
                tri[2].varyings[1] = flat;
            }
        }

        // 后续管线阶段
        let primitives = self.primitive_assembly(primitive_array);
        let mut fragment_inputs: Vec<FragmentInput> = Vec::new();
        for primitive in &primitives {
            if let Some(mut fragments) = self.rasterization(primitive) {
                fragment_inputs.append(&mut fragments);
            }
        }

        if let Some(filtered_frags) = self.depth_test(fragment_inputs) {
            for frag in filtered_frags {
                let color = (self.fragment_shader)(uniforms, &frag);
                self.raster_operations(frag.pos.x as usize, frag.pos.y as usize, color);
            }
        }
    }



    fn primitive_assembly(&self, input: Vec<[VertexOutput; 3]>) -> Vec<PrimitiveOutput> {
        let mut out: Vec<PrimitiveOutput> = Vec::new();
        for item in input {
            out.push(PrimitiveOutput { triangle: item });
        }
        out
    }

    fn rasterization(&mut self, input: &PrimitiveOutput) -> Option<Vec<FragmentInput>> {
        let w = self.w as f32;
        let h = self.h as f32;

        let mut fragments: Vec<FragmentInput> = Vec::new();

        // clip-space → NDC（透视除法）→ 屏幕空间
        let to_screen = |pos: &Vec4| -> Option<IVec2> {
            if pos.w.abs() < 1e-8 {
                return None; // 剔除 w 接近 0 的退化三角形
            }
            let ndc = pos.truncate() / pos.w; // 透视除法 → NDC [-1,1]
            Some(IVec2::new(
                ((ndc.x + 1.0) * 0.5 * w - 0.5).floor() as i32,
                ((1.0 - ndc.y) * 0.5 * h - 0.5).floor() as i32,
            ))
        };

        let p0 = to_screen(&input.triangle[0].pos);
        let p1 = to_screen(&input.triangle[1].pos);
        let p2 = to_screen(&input.triangle[2].pos);

        let w0_clip = input.triangle[0].pos.w;
        let w1_clip = input.triangle[1].pos.w;
        let w2_clip = input.triangle[2].pos.w;

        let d0 = input.triangle[0].pos.z;
        let d1 = input.triangle[1].pos.z;
        let d2 = input.triangle[2].pos.z;

        let varyings0 = &input.triangle[0].varyings;
        let varyings1 = &input.triangle[1].varyings;
        let varyings2 = &input.triangle[2].varyings;

        if let (Some(p0), Some(p1), Some(p2)) = (p0, p1, p2) {
            match self.polygon_mode {
                PolygonMode::FILL => {
                    // DrawTriangleFill::draw(self.framebuffer, &p0, &p1, &p2, &tgaimage::RED);
                    // 计算包围盒，并裁剪到屏幕范围内
                    let w = self.w as i32;
                    let h = self.h as i32;
                    let x_min = min_3(p0.x, p1.x, p2.x).clamp(0, w - 1);
                    let x_max = max_3(p0.x, p1.x, p2.x).clamp(0, w - 1);
                    let y_min = min_3(p0.y, p1.y, p2.y).clamp(0, h - 1);
                    let y_max = max_3(p0.y, p1.y, p2.y).clamp(0, h - 1);

                    // 判断包围盒里像素是在三角形内还是外
                    let eps = 1e-6f32;
                    let p0_f = Vec2::new(p0.x as f32, p0.y as f32);
                    let p1_f = Vec2::new(p1.x as f32, p1.y as f32);
                    let p2_f = Vec2::new(p2.x as f32, p2.y as f32);
                    let area = (p1_f - p0_f).perp_dot(p2_f - p0_f);

                    if area.abs() < eps {
                        return None;
                    }

                    let is_front_face = area < 0.0;
                    let should_cull = match self.cull {
                        CullMode::BACK => !is_front_face,
                        CullMode::FRONT => is_front_face,
                        CullMode::NULL => false,
                    };
                    if should_cull {
                        return None;
                    }

                    for x in x_min..=x_max {
                        for y in y_min..=y_max {
                            let p_center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                            let w0 = ((p2_f - p1_f).perp_dot(p_center - p1_f)) / area;
                            let w1 = ((p0_f - p2_f).perp_dot(p_center - p2_f)) / area;
                            let w2 = ((p1_f - p0_f).perp_dot(p_center - p0_f)) / area;

                            if w0 >= -eps && w1 >= -eps && w2 >= -eps {
                                let inv_w0 = 1.0 / w0_clip;
                                let inv_w1 = 1.0 / w1_clip;
                                let inv_w2 = 1.0 / w2_clip;

                                let denom = w0 * inv_w0 + w1 * inv_w1 + w2 * inv_w2;
                                let r1 = (w0 * inv_w0) / denom;
                                let r2 = (w1 * inv_w1) / denom;
                                let r3 = (w2 * inv_w2) / denom;

                                let depth = r1 * d0 + r2 * d1 + r3 * d2;

                                let interpolated_varyings = interpolate_varyings(
                                    &varyings0, &varyings1, &varyings2, r1, r2, r3,
                                );

                                let frag = FragmentInput {
                                    pos: IVec2 { x, y },
                                    depth,
                                    varyings: interpolated_varyings,
                                };
                                fragments.push(frag);
                            }
                        }
                    }

                    return Some(fragments);
                }
                PolygonMode::LINE => {
                    // LINE mode using generic framebuffer
                    let red = TGAColor::new(1.0, 0.0, 0.0, 1.0);
                    draw_line(self.framebuffer, &p0, &p1, &red);
                    draw_line(self.framebuffer, &p1, &p2, &red);
                    draw_line(self.framebuffer, &p0, &p2, &red);
                }
                PolygonMode::Point => {}
            }
        }
        return None;
    }

    fn depth_test(&mut self, frags: Vec<FragmentInput>) -> Option<Vec<FragmentInput>> {
        let w = self.w;
        let h = self.h;
        let depth_buffer = &mut self.depth_buffer;
        let result: Vec<FragmentInput> = frags
            .into_iter()
            .filter(|frag| {
                let x = frag.pos.x as usize;
                let y = frag.pos.y as usize;
                // 跳过屏幕外的片段
                if x >= w || y >= h {
                    return false;
                }
                let idx = y * w + x;
                if frag.depth < depth_buffer[idx] {
                    depth_buffer[idx] = frag.depth;
                    true
                } else {
                    false
                }
            })
            .collect();

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    fn raster_operations(&mut self, x: usize, y: usize, color: TGAColor) {
        // mix color
        let src_color = self.framebuffer.get(x, y);
        let dst_color = match src_color {
            Some(s_color) => {
                TGAColor::new(
                    color.a * color.r + (1.0 - color.a) * s_color.r,
                    color.a * color.g + (1.0 - color.a) * s_color.g,
                    color.a * color.b + (1.0 - color.a) * s_color.b,
                    1.0
                )
            }
            None => {
                TGAColor::new(color.r, color.g, color.b, 1.0)
            }
        };
        self.framebuffer.set(x, y, &dst_color);
    }

}

// 输入: 一个顶点的原始属性
#[derive(Clone)]
pub struct VertexInput {
    pub pos: Vec3,
    pub varyings: Vec<Varying>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framebuffer::FrameBuffer;

    /// 构造一个覆盖屏幕中心的大三角形，法线朝 +z（相机方向）
    fn fullscreen_triangle() -> Vec<VertexInput> {
        let white = TGAColor::new(1.0, 1.0, 1.0, 1.0);
        vec![
            VertexInput {
                pos: Vec3::new(-0.8, -0.8, 0.0),
                varyings: vec![
                    Varying::Color(white),
                    Varying::Vec3(Vec3::new(0.0, 0.0, 1.0)),
                    Varying::Vec2(Vec2::new(0.5, 0.5)),
                ],
            },
            VertexInput {
                pos: Vec3::new(0.8, -0.8, 0.0),
                varyings: vec![
                    Varying::Color(white),
                    Varying::Vec3(Vec3::new(0.0, 0.0, 1.0)),
                    Varying::Vec2(Vec2::new(0.5, 0.5)),
                ],
            },
            VertexInput {
                pos: Vec3::new(0.0, 0.8, 0.0),
                varyings: vec![
                    Varying::Color(white),
                    Varying::Vec3(Vec3::new(0.0, 0.0, 1.0)),
                    Varying::Vec2(Vec2::new(0.5, 0.5)),
                ],
            },
        ]
    }

    /// light/view 都沿 +z：half 向量 = +z，与法线完全对齐 → Blinn-Phong 高光最强
    fn light_uniforms(specular_color: Vec3) -> Uniforms<'static> {
        Uniforms {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            model_view: Mat4::IDENTITY,
            model_view_proj: Mat4::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
            light_dir: Vec3::new(0.0, 0.0, 1.0),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            // ambient 置零：断言不依赖环境光强度常量
            ambient_color: Vec3::ZERO,
            diffuse_color: Vec3::new(0.7, 0.7, 0.7),
            specular_color,
            diffuse_tex: None,
            normal_tex: None,
            specular_tex: None,
            glossiness_tex: None,
        }
    }

    #[test]
    fn blinn_phong_specular_produces_highlight() {
        // 有高光：ambient(0) + diffuse(0.7) + specular(0.3) = 1.0
        let mut fb = FrameBuffer::new(100, 100);
        let mut pipeline = RenderPipleline::new(&mut fb);
        pipeline.set_draw_mode(PolygonMode::FILL);
        pipeline.draw(
            &fullscreen_triangle(),
            &light_uniforms(Vec3::new(0.3, 0.3, 0.3)),
        );
        let lit = fb.get(50, 50).unwrap();
        assert!(lit.r > 0.95, "高光区域应接近白色, got {}", lit.r);

        // 无高光：只有 ambient(0) + diffuse(0.7) = 0.7
        let mut fb2 = FrameBuffer::new(100, 100);
        let mut pipeline2 = RenderPipleline::new(&mut fb2);
        pipeline2.set_draw_mode(PolygonMode::FILL);
        pipeline2.draw(&fullscreen_triangle(), &light_uniforms(Vec3::ZERO));
        let unlit = fb2.get(50, 50).unwrap();
        assert!(
            (unlit.r - 0.7).abs() < 0.02,
            "无高光时应约为 0.7, got {}",
            unlit.r
        );
        assert!(lit.r > unlit.r + 0.1, "高光应让像素明显更亮");
    }
}

#[derive(Clone, Debug)]
pub struct VertexOutput {
    pub pos: Vec4,
    pub varyings: Vec<Varying>,
}

#[derive(Clone, Copy, Debug)]
pub enum Varying {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Color(TGAColor),
}

impl Varying {
    fn interpolate(a: Varying, b: Varying, c: Varying, r1: f32, r2: f32, r3: f32) -> Varying {
        match (a, b, c) {
            (Varying::Float(a), Varying::Float(b), Varying::Float(c)) => {
                Varying::Float(a * r1 + b * r2 + c * r3)
            }
            (Varying::Vec2(a), Varying::Vec2(b), Varying::Vec2(c)) => {
                Varying::Vec2(a * r1 + b * r2 + c * r3)
            }
            (Varying::Vec3(a), Varying::Vec3(b), Varying::Vec3(c)) => {
                Varying::Vec3(a * r1 + b * r2 + c * r3)
            }
            (Varying::Vec4(a), Varying::Vec4(b), Varying::Vec4(c)) => {
                Varying::Vec4(a * r1 + b * r2 + c * r3)
            }
            (Varying::Color(a), Varying::Color(b), Varying::Color(c)) => {
                Varying::Color(TGAColor::new(
                    a.r * r1 + b.r * r2 + c.r * r3,
                    a.g * r1 + b.g * r2 + c.g * r3,
                    a.b * r1 + b.b * r2 + c.b * r3,
                    a.a * r1 + b.a * r2 + c.a * r3,
                ))
            }
            _ => unreachable!("interpolation type mismatch"),
        }
    }
}

fn interpolate_varyings(
    varyings0: &[Varying],
    varyings1: &[Varying],
    varyings2: &[Varying],
    r1: f32,
    r2: f32,
    r3: f32,
) -> Vec<Varying> {
    varyings0
        .iter()
        .zip(varyings1.iter())
        .zip(varyings2.iter())
        .map(|((a, b), c)| Varying::interpolate(*a, *b, *c, r1, r2, r3))
        .collect()
}

/// 使用 Bresenham 算法在泛型帧缓冲上绘制线段
fn draw_line(target: &mut dyn FrameBufferTarget, p0: &IVec2, p1: &IVec2, color: &TGAColor) {
    let mut x0 = p0.x;
    let mut y0 = p0.y;
    let mut x1 = p1.x;
    let mut y1 = p1.y;

    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    if steep {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let dy_abs = dy.abs();
    let y_step: i32 = if y1 >= y0 { 1 } else { -1 };
    let mut d = 2 * dy_abs - dx;
    let mut y = y0;

    for x in x0..=x1 {
        if steep {
            target.set(y as usize, x as usize, color);
        } else {
            target.set(x as usize, y as usize, color);
        }
        if d > 0 {
            y += y_step;
            d += 2 * (dy_abs - dx);
        } else {
            d += 2 * dy_abs;
        }
    }
}

#[derive(Default)]
pub struct FragmentInput {
    pub pos: IVec2,
    pub depth: f32,
    pub varyings: Vec<Varying>,
}

struct PrimitiveOutput {
    triangle: [VertexOutput; 3],
}

pub struct Uniforms<'a> {
    // ---- 变换矩阵 ----
    pub model: Mat4,           // 模型矩阵 (世界变换)
    pub view: Mat4,            // 视图矩阵 (相机变换)
    pub projection: Mat4,      // 投影矩阵 (透视/正交)
    pub model_view: Mat4,      // 预乘: view * model
    pub model_view_proj: Mat4, // 预乘: projection * view * model
    pub normal_matrix: Mat3,   // 法线变换矩阵 (MVT 的逆的转置)

    // ---- 材质参数 ----
    pub light_dir: Vec3,      // 光照方向
    pub view_dir: Vec3,       // 视角方向
    pub ambient_color: Vec3,  // 环境光颜色
    pub diffuse_color: Vec3,  // 漫反射颜色
    pub specular_color: Vec3, // 高光颜色

    // ---- 纹理 ----
    pub diffuse_tex: Option<&'a TGAImage>,    // 漫反射纹理
    pub normal_tex: Option<&'a TGAImage>,     // 法线纹理
    pub specular_tex: Option<&'a TGAImage>,   // 高光贴图
    pub glossiness_tex: Option<&'a TGAImage>, // 光泽度贴图
}
