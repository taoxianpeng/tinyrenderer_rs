pub use crate::drawline::{ DDA, Bresenham };
pub use crate::tgaimage::{TGAColor, TGAImage, TGAImageType};
pub use crate::datatype::Point2D;
pub use glam::Vec2;

// -------- 浮点 DDA 版 (新) --------

pub struct DrawTriangleFloat;

impl DrawTriangleFloat {
    pub fn draw(
        image: &mut TGAImage,
        p0: &Vec2,
        p1: &Vec2,
        p2: &Vec2,
        c: &TGAColor,
    ) {
        DDA::draw(image, p0, p1, c);
        DDA::draw(image, p1, p2, c);
        DDA::draw(image, p0, p2, c);
    }
}


pub struct DrawTriangle {}

impl DrawTriangle {
    pub fn draw(
        image: &mut TGAImage,
        p0: &Point2D,
        p1: &Point2D,
        p2: &Point2D,
        c: &TGAColor,
    ) {
        Bresenham::draw(image, p0, p1, c);
        Bresenham::draw(image, p1, p2, c);
        Bresenham::draw(image, p0, p2, c);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tri_1() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });

        let p0 = Point2D { x: 250, y: 50 };
        let p1 = Point2D { x: 50, y: 450 };
        let p2 = Point2D { x: 450, y: 450 };

        DrawTriangle::draw(
            &mut image,
            &p0,
            &p1,
            &p2,
            &TGAColor {
                r: 100,
                g: 100,
                b: 29,
                a: 255,
            },
        );

        image
            .write_tga_file("output_triangle.tga", false, true)
            .unwrap();
    }
}

// -------- 新 DDA 三角形测试 --------

#[cfg(test)]
mod test_float {
    use super::*;

    #[test]
    fn test_tri_dda_basic() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&TGAColor { r: 255, g: 255, b: 255, a: 255 });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(250.0, 50.0),
            &Vec2::new(50.0, 450.0),
            &Vec2::new(450.0, 450.0),
            &TGAColor { r: 100, g: 100, b: 29, a: 255 },
        );

        image.write_tga_file("output_triangle.tga", false, true).unwrap();
    }

    #[test]
    fn test_tri_dda_flat_line() {
        // 三个点近乎共线，浮点端点
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor { r: 255, g: 255, b: 255, a: 255 });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(20.3, 100.7),
            &Vec2::new(100.0, 100.5),
            &Vec2::new(180.8, 100.2),
            &TGAColor { r: 50, g: 150, b: 200, a: 255 },
        );

        image.write_tga_file("output_triangle.tga", false, true).unwrap();
    }

    #[test]
    fn test_tri_dda_steep() {
        // 瘦高三角形（陡峭边）
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor { r: 255, g: 255, b: 255, a: 255 });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(100.0, 20.0),
            &Vec2::new(30.0, 180.0),
            &Vec2::new(170.0, 180.0),
            &TGAColor { r: 200, g: 50, b: 80, a: 255 },
        );

        image.write_tga_file("output_triangle.tga", false, true).unwrap();
    }

    #[test]
    fn test_tri_dda_tiny() {
        // 小三角形
        let mut image = TGAImage::new(20, 20, TGAImageType::RGB);
        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(3.7, 4.2),
            &Vec2::new(15.1, 6.8),
            &Vec2::new(8.3, 16.5),
            &TGAColor { r: 255, g: 0, b: 0, a: 255 },
        );

        image.write_tga_file("output_triangle.tga", false, true).unwrap();
    }
}
