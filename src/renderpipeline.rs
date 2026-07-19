use crate::drawline::TGAColor;
use crate::drawtriangle::DrawTriangle;
use crate::renderpipeline::{ProjectionMode::ORTHO, ProjectionMode::PERSPECTIVE};
use crate::tgaimage;
use crate::tgaimage::TGAImage;
use glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4, vec4};
use std::cmp::{max, min};
use std::vec;

pub enum PolygonMode {
    FILL,
    LINE,
    Point,
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

pub struct RenderPipleline<'a> {
    buffer: Option<&'a Vec<VertexInput>>,
    uniforms: Option<&'a Uniforms<'a>>,
    polygon_mode: PolygonMode,
    flat_normal: bool,
    framebuffer: &'a mut TGAImage,
    w: usize,
    h: usize,
    depth_buffer: Vec<f32>,
    color_buffer: Vec<TGAColor>,
}

impl<'a> RenderPipleline<'a> {
    pub fn new(framebuffer: &'a mut TGAImage) -> RenderPipleline<'a> {
        let w = framebuffer.width();
        let h = framebuffer.height();
        let total_pixels = w * h;
        RenderPipleline {
            buffer: None,
            uniforms: None,
            polygon_mode: PolygonMode::LINE,
            flat_normal: false,
            framebuffer,
            w,
            h,
            depth_buffer: vec![f32::MAX; total_pixels],
            color_buffer: vec![TGAColor::new(0.0, 0.0, 0.0, 0.0); total_pixels],
        }
    }

    pub fn add_data(&mut self, data: &'a Vec<VertexInput>) {
        self.buffer = Some(data);
    }

    pub fn remove_data(&mut self) {
        self.buffer = None;
    }

    pub fn set_draw_mode(&mut self, mode: PolygonMode) {
        self.polygon_mode = mode;
    }

    pub fn set_uniforms(&mut self, unforms: &'a Uniforms) {
        self.uniforms = Some(unforms);
    }

    pub fn set_flat_normal(&mut self, enable: bool) {
        self.flat_normal = enable;
    }

    pub fn draw(&mut self) {
        if let (Some(vertex_array), Some(uniforms)) = (self.buffer, self.uniforms) {
            let mut primitive_array: Vec<[VertexOutput; 3]> = Vec::new();

            // 每 3 个顶点为一组（一个三角形），执行 vertex shader
            for chunk in vertex_array.chunks(3) {
                if chunk.len() < 3 {
                    break; // 不足 3 个顶点，丢弃
                }

                let v0 = self.vertex_shader(&chunk[0], uniforms);
                let v1 = self.vertex_shader(&chunk[1], uniforms);
                let v2 = self.vertex_shader(&chunk[2], uniforms);

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
                self.fragment_shader(uniforms, filtered_frags);
            }

            self.raster_operations();
        }
    }

    fn vertex_shader(&self, input: &VertexInput, uniforms: &Uniforms) -> VertexOutput {
        let pos = uniforms.model_view_proj * input.pos.extend(1.0);
        let varyings = input.varyings.clone();
        VertexOutput { pos, varyings }
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
                    // 计算包围盒
                    let x_min = min_3(p0.x, p1.x, p2.x);
                    let x_max = max_3(p0.x, p1.x, p2.x);
                    let y_min = min_3(p0.y, p1.y, p2.y);
                    let y_max = max_3(p0.y, p1.y, p2.y);

                    // 判断包围盒里像素是在三角形内还是外
                    let eps = 1e-6f32;
                    let p0_f = Vec2::new(p0.x as f32, p0.y as f32);
                    let p1_f = Vec2::new(p1.x as f32, p1.y as f32);
                    let p2_f = Vec2::new(p2.x as f32, p2.y as f32);
                    let area = (p1_f - p0_f).perp_dot(p2_f - p0_f);

                    if area.abs() < eps {
                        return None;
                    }

                    for x in x_min..=x_max {
                        for y in y_min..=y_max {
                            let p_center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                            let w0 = ((p1_f - p0_f).perp_dot(p_center - p0_f)) / area;
                            let w1 = ((p2_f - p1_f).perp_dot(p_center - p1_f)) / area;
                            let w2 = ((p0_f - p2_f).perp_dot(p_center - p2_f)) / area;

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
                    // DrawTriangleFloat::draw(self.framebuffer, &p0, &p1, &p2, &tgaimage::RED);
                    DrawTriangle::draw(self.framebuffer, &p0, &p1, &p2, &tgaimage::RED);
                }
                PolygonMode::Point => {}
            }
        }
        return None;
    }

    fn depth_test(&mut self, frags: Vec<FragmentInput>) -> Option<Vec<FragmentInput>> {
        let w = self.w;
        let depth_buffer = &mut self.depth_buffer;
        let result: Vec<FragmentInput> = frags
            .into_iter()
            .filter(|frag| {
                let idx = frag.pos.y as usize * w + frag.pos.x as usize;
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

    fn fragment_shader(&mut self, uniforms: &Uniforms, frags: Vec<FragmentInput>) {
        for frag in frags {
            let i = frag.pos.y as usize * self.w + frag.pos.x as usize;

            let color = match frag.varyings[0] {
                Varying::Color(color) => color,
                _ => unreachable!("first varying must be color"),
            };
            let normal = match frag.varyings[1] {
                Varying::Vec3(normal) => normal,
                _ => unreachable!("second varying must be normal"),
            };

            let ambient_light_strength = 0.1;
            let ambient = ambient_light_strength * uniforms.ambient_color;

            let diff = f32::max(normal.dot(uniforms.light_dir), 0.0);
            let diffuse = diff * uniforms.ambient_color;

            let specular_light_strength = 1.0;
            let halfway_dir = (uniforms.light_dir + uniforms.view_dir).normalize();
            let spec = f32::powi(f32::max(normal.dot(halfway_dir), 0.0), 32);
            let specular = specular_light_strength * spec * uniforms.ambient_color;

            let rate = ambient + diffuse + specular;

            let lit_color = TGAColor::new(
                rate.x * color.r,
                rate.y * color.g,
                rate.z * color.b,
                color.a,
            );
            self.color_buffer[i] = lit_color;
        }
    }

    fn raster_operations(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                let index = y * self.w + x;
                let c = self.color_buffer[index];
                self.framebuffer.set(x, y, &c);
            }
        }
    }
}

// 输入: 一个顶点的原始属性
pub struct VertexInput {
    pub pos: Vec3,
    pub varyings: Vec<Varying>,
}

#[derive(Clone, Debug)]
struct VertexOutput {
    pos: Vec4,
    varyings: Vec<Varying>,
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

#[derive(Default)]
struct FragmentInput {
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
