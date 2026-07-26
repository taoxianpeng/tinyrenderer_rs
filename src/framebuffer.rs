use crate::tgaimage::{TGAColor, TGAImage, TGAImageType};

/// 抽象帧缓冲接口，渲染管线可同时对接 FrameBuffer 和 TGAImage
pub trait FrameBufferTarget {
    fn set(&mut self, x: usize, y: usize, color: &TGAColor);
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}

pub struct FrameBuffer {
    buffer: Vec<u32>,
    width: usize,
    height: usize,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> FrameBuffer {
        FrameBuffer {
            buffer: vec![0; width * height],
            width,
            height,
        }
    }

    pub fn set_buffer(&mut self, buffer: Vec<u32>) {
        self.buffer = buffer;
    }

    /// 用指定颜色填充整个帧缓冲
    pub fn clear_color(&mut self, color: &TGAColor) {
        let r = (color.r.clamp(0.0, 1.0) * 255.0).round() as u32;
        let g = (color.g.clamp(0.0, 1.0) * 255.0).round() as u32;
        let b = (color.b.clamp(0.0, 1.0) * 255.0).round() as u32;
        let pixel = (r << 16) | (g << 8) | b;
        self.buffer.fill(pixel);
    }

    /// 设置 (x, y) 处的像素颜色
    pub fn set(&mut self, x: usize, y: usize, color: &TGAColor) {
        if x >= self.width || y >= self.height {
            return;
        }
        let r = (color.r.clamp(0.0, 1.0) * 255.0).round() as u32;
        let g = (color.g.clamp(0.0, 1.0) * 255.0).round() as u32;
        let b = (color.b.clamp(0.0, 1.0) * 255.0).round() as u32;
        self.buffer[x + y * self.width] = (r << 16) | (g << 8) | b;
    }

    /// 获取 (x, y) 处的像素颜色，越界返回 None
    pub fn get(&self, x: usize, y: usize) -> Option<TGAColor> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let pixel = self.buffer[x + y * self.width];
        let r = ((pixel >> 16) & 0xFF) as f32 / 255.0;
        let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
        let b = (pixel & 0xFF) as f32 / 255.0;
        Some(TGAColor {
            r,
            g,
            b,
            a: 1.0,
        })
    }

    pub fn as_buffer(&self) -> &[u32] {
        &self.buffer
    }

    pub fn save_to_image(&self, path: &str) {
        let mut tgaimg = TGAImage::new(self.width, self.height, TGAImageType::RGB);
        // minifb 使用 0x00RRGGBB 格式，TGA RGB 按 B, G, R 字节顺序存储
        tgaimg.data = self
            .buffer
            .iter()
            .flat_map(|&pixel| {
                let b = (pixel & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let r = ((pixel >> 16) & 0xFF) as u8;
                [b, g, r]
            })
            .collect();
        tgaimg.write_tga_file(path, false, true).unwrap();
        println!("输出完成: {} ({}x{})", path, self.width, self.height);
    }
}

impl FrameBufferTarget for FrameBuffer {
    fn set(&mut self, x: usize, y: usize, color: &TGAColor) {
        self.set(x, y, color);
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }
}

impl FrameBufferTarget for TGAImage {
    fn set(&mut self, x: usize, y: usize, color: &TGAColor) {
        self.set(x, y, color);
    }

    fn width(&self) -> usize {
        self.width()
    }

    fn height(&self) -> usize {
        self.height()
    }
}