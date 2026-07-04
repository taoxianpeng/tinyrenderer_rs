use crate::drawtriangle::DrawTriangleFloat;
use crate::tgaimage;
use crate::tgaimage::TGAImage;
use glam::Mat3;
use glam::Mat4;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;

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

pub fn lookat() -> Mat4 {
    Mat4::IDENTITY
}

pub fn perspective() -> Mat4 {
    Mat4::IDENTITY
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
