use std::vec;
use std::cmp::{max, min};
use crate::drawline::TGAColor;
use crate::drawtriangle::DrawTriangle;
use crate::renderpipeline::{ ProjectionMode::ORTHO, ProjectionMode::PERSPECTIVE };
use crate::tgaimage;
use crate::tgaimage::TGAImage;
use glam::{ Mat3, Mat4, Vec2, IVec2, Vec3, Vec4, vec4};

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
    framebuffer: &'a mut TGAImage,
    w: usize, 
    h: usize,
    depth_buffer: Vec<f32>,
    color_buffer: Vec<TGAColor>,
}

impl<'a> RenderPipleline<'a> {
    pub fn new(framebuffer: &'a mut TGAImage) -> RenderPipleline {
        let w = framebuffer.width();
        let h = framebuffer.height();
        let total_pixels = w * h;
        RenderPipleline {
            buffer: None,
            uniforms: None,
            polygon_mode: PolygonMode::LINE,
            framebuffer,
            w,
            h,
            depth_buffer: vec![f32::MAX; total_pixels],
            color_buffer: vec![TGAColor::new(0, 0, 0, 0); total_pixels],
        }
    }

    pub fn add_data(&mut self, data: &'a Vec<VertexInput>) {
        self.buffer = Some(data);
    }

    pub fn remove_data(&mut self) {
        self.buffer = None;
    }

    pub fn set_draw_mode(&mut self,  mode: PolygonMode) {
        self.polygon_mode = mode;
    }

    pub fn set_uniforms(&mut self, unforms: &'a Uniforms) {
        self.uniforms = Some(unforms);
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
        VertexOutput {
            position: uniforms.model_view_proj * input.position.extend(1.0),
            color: input.color,
            normal: input.normal,
            texcoord: input.texcoord,
        }
    }

    fn primitive_assembly(&self, input: Vec<[VertexOutput; 3]>) -> Vec<PrimitiveOutput> {
        let mut out: Vec<PrimitiveOutput> = Vec::new();
        for item in input {
            out.push(PrimitiveOutput { triangle: item });
        }
        out
    }

    fn rasterization(&mut self, input: &PrimitiveOutput) -> Option<Vec<FragmentInput>>{
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
                ((ndc.x + 1.0) * 0.5 * w).round() as i32, // x: [-1,1] → [0,width]
                ((1.0 - ndc.y) * 0.5 * h).round() as i32, // y: [-1,1] → [0,height]，翻转 Y
            ))
        };

        let p0 = to_screen(&input.triangle[0].position);
        let p1 = to_screen(&input.triangle[1].position);
        let p2 = to_screen(&input.triangle[2].position);

        let clr0 = &input.triangle[0].color;
        let clr1 = &input.triangle[1].color;
        let clr2 = &input.triangle[2].color;

        let d0 = &input.triangle[0].position.z;
        let d1 = &input.triangle[1].position.z;
        let d2 = &input.triangle[2].position.z;

        let n0 = &input.triangle[0].normal;
        let n1 = &input.triangle[1].normal;
        let n2 = &input.triangle[2].normal;

        let tc0 = &input.triangle[0].texcoord;
        let tc1 = &input.triangle[1].texcoord;
        let tc2 = &input.triangle[2].texcoord;

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
                        for x in x_min..=x_max {
                            for y in y_min..=y_max {
                                let p = IVec2::new(x, y);
                                let c1 = (p1 - p0).perp_dot(p - p0);
                                let c2 = (p2 - p1).perp_dot(p - p1);
                                let c3 = (p0 - p2).perp_dot(p - p2);

                                let mut render_flag = false;
                                if (c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0) {
                                    // image.set(p.x as usize, p.y as usize, c);
                                    render_flag = true;
                                }

                                if c1 == 0 && is_top_left_edge(&p0, &p1) && is_in_edge(&p, &p0, &p1) {
                                    render_flag = true;
                                } else if c2 == 0 && is_top_left_edge(&p1, &p2) && is_in_edge(&p, &p1, &p2)
                                {
                                    render_flag = true;
                                } else if c3 == 0 && is_top_left_edge(&p2, &p0) && is_in_edge(&p, &p2, &p0)
                                {
                                    render_flag = true;
                                }

                                if render_flag {
                                    let aren = (c1 + c2 + c3) as f32;
                                    let r1 =  (c2 as f32) / aren;
                                    let r2 =  (c3 as f32) / aren;
                                    let r3 =  (c1 as f32) / aren;

                                    let color = TGAColor::new(
                                        (r1 * clr0.r as f32 + r2 * clr1.r as f32 + r3 * clr2.r as f32) as u8,
                                        (r1 * clr0.g as f32 + r2 * clr1.g as f32 + r3 * clr2.g as f32) as u8,
                                        (r1 * clr0.b as f32 + r2 * clr1.b as f32 + r3 * clr2.b as f32) as u8,
                                        (r1 * clr0.a as f32 + r2 * clr1.a as f32 + r3 * clr2.a as f32) as u8,
                                    );

                                    let depth = r1 * d0 + r2 * d1 + r3 * d2;
                                    let normal = r1 * *n0 + r2 * *n1 + r3 * *n2;
                                    let texcoord = r1 * *tc0 + r2 * *tc1 + r3 * *tc2;

                                    // self.framebuffer.set(x as usize, y as usize, &color);
                                    let frag = FragmentInput {
                                        pos: IVec2 { x, y },
                                        depth,
                                        color,
                                        normal,
                                        texcoord,
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
        return None
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

    fn fragment_shader(&mut self, _uniforms: &Uniforms, frags: Vec<FragmentInput>) {
        for frag in frags {
            let i = frag.pos.y as usize * self.w + frag.pos.x as usize;
            self.color_buffer[i] = frag.color;
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
    pub position: Vec3,
    pub color: TGAColor,
    pub normal: Vec3,
    pub texcoord: Vec2,
}

// 输出: 变换后的顶点数据，传递给下一个阶段
struct VertexOutput {
    position: Vec4, // clip-space 位置 (必须)
    color: TGAColor,
    normal: Vec3,
    texcoord: Vec2,
}

#[derive(Default)]
struct FragmentInput {
    pub pos: IVec2,
    pub depth: f32,
    pub color: TGAColor,
    pub normal: Vec3,
    pub texcoord: Vec2,
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
    pub camera_pos: Vec3,     // 相机位置
    pub ambient_color: Vec3,  // 环境光颜色
    pub diffuse_color: Vec3,  // 漫反射颜色
    pub specular_color: Vec3, // 高光颜色

    // ---- 纹理 ----
    pub diffuse_tex: Option<&'a TGAImage>,    // 漫反射纹理
    pub normal_tex: Option<&'a TGAImage>,     // 法线纹理
    pub specular_tex: Option<&'a TGAImage>,   // 高光贴图
    pub glossiness_tex: Option<&'a TGAImage>, // 光泽度贴图
}