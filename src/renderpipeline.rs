use std::vec;

use crate::drawtriangle::DrawTriangleFloat;
use crate::renderpipeline::ProjectionMode::ORTHO;
use crate::renderpipeline::ProjectionMode::PERSPECTIVE;
use crate::tgaimage;
use crate::tgaimage::TGAImage;
use glam::Mat3;
use glam::Mat4;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;
use glam::vec4;

pub enum PolygonMode {
    FILL,
    LINE,
    Point,
}

pub struct RenderPipleline<'a> {
    buffer: Option<&'a Vec<VertexInput>>,
    uniforms: Option<&'a Uniforms<'a>>,
    polygon_mode: PolygonMode,
    framebuffer: &'a mut TGAImage,
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
            let m_33 = (z_near + z_far) / (z_near - z_far);
            let m_34 = -2.0 * z_near * z_far / (z_near - z_far);

            Mat4::from_cols(
                vec4(1.0 / (aspect_ratio * tan_fov_div_2), 0.0, 0.0, 0.0),
                vec4(0.0, 1.0 / tan_fov_div_2, 0.0, 0.0),
                vec4(0.0, 0.0, m_33, -1.0),
                vec4(0.0, 0.0, m_34, 0.0),
            )
        }
    }
}

impl<'a> RenderPipleline<'a> {
    pub fn new(framebuffer: &'a mut TGAImage) -> RenderPipleline {
        RenderPipleline {
            buffer: None,
            uniforms: None,
            polygon_mode: PolygonMode::LINE,
            framebuffer: framebuffer,
        }
    }

    pub fn add_data(&mut self, data: &'a Vec<VertexInput>) {
        self.buffer = Some(data);
    }

    pub fn remove_data(&mut self) {
        self.buffer = None;
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
            for primitive in &primitives {
                self.rasterization(primitive);
            }

            if let Some(uniforms_) = self.uniforms {
                self.fragment_shader(uniforms_);
            }

            self.depth_test();
        }
    }

    fn vertex_shader(&self, input: &VertexInput, uniforms: &Uniforms) -> VertexOutput {
        VertexOutput {
            position: uniforms.model_view_proj * input.position.extend(1.0),
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

    fn rasterization(&mut self, input: &PrimitiveOutput) {
        let w = self.framebuffer.width() as f32;
        let h = self.framebuffer.height() as f32;

        // clip-space → NDC（透视除法）→ 屏幕空间
        let to_screen = |pos: &Vec4| -> Option<Vec2> {
            if pos.w.abs() < 1e-8 {
                return None; // 剔除 w 接近 0 的退化三角形
            }
            let ndc = pos.truncate() / pos.w; // 透视除法 → NDC [-1,1]
            Some(Vec2::new(
                (ndc.x + 1.0) * 0.5 * w, // x: [-1,1] → [0,width]
                (1.0 - ndc.y) * 0.5 * h, // y: [-1,1] → [0,height]，翻转 Y
            ))
        };

        match self.polygon_mode {
            PolygonMode::FILL => {}
            PolygonMode::LINE => {
                let p0 = to_screen(&input.triangle[0].position);
                let p1 = to_screen(&input.triangle[1].position);
                let p2 = to_screen(&input.triangle[2].position);
                if let (Some(p0), Some(p1), Some(p2)) = (p0, p1, p2) {
                    DrawTriangleFloat::draw(self.framebuffer, &p0, &p1, &p2, &tgaimage::RED);
                }
            }
            PolygonMode::Point => {}
        }
    }

    fn fragment_shader(&mut self, uniforms: &Uniforms) {}

    fn depth_test(&mut self) {}
}

// 输入: 一个顶点的原始属性
pub struct VertexInput {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
}

// 输出: 变换后的顶点数据，传递给下一个阶段
struct VertexOutput {
    position: Vec4, // clip-space 位置 (必须)
    normal: Vec3,
    texcoord: Vec2,
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
