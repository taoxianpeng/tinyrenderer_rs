use std::cmp::{max, min};

use crate::datatype::Point2D;
pub use crate::drawline::{Bresenham, DDA};
pub use crate::tgaimage::{TGAColor, TGAImage, TGAImageType};
use glam::{IVec2, Vec2};

pub struct DrawTriangleFloat;

impl DrawTriangleFloat {
    pub fn draw(image: &mut TGAImage, p0: &Vec2, p1: &Vec2, p2: &Vec2, c: &TGAColor) {
        DDA::draw(image, p0, p1, c);
        DDA::draw(image, p1, p2, c);
        DDA::draw(image, p0, p2, c);
    }
}

pub struct DrawTriangle {}

impl DrawTriangle {
    pub fn draw(image: &mut TGAImage, p0: &Point2D, p1: &Point2D, p2: &Point2D, c: &TGAColor) {
        Bresenham::draw(image, p0, p1, c);
        Bresenham::draw(image, p1, p2, c);
        Bresenham::draw(image, p0, p2, c);
    }
}

pub struct DrawTriangleFill {}

impl DrawTriangleFill {
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

    pub fn draw(image: &mut TGAImage, p0: &IVec2, p1: &IVec2, p2: &IVec2, c: &TGAColor) {
        // 计算包围盒
        let x_min = Self::min_3(p0.x, p1.x, p2.x);
        let x_max = Self::max_3(p0.x, p1.x, p2.x);
        let y_min = Self::min_3(p0.y, p1.y, p2.y);
        let y_max = Self::max_3(p0.y, p1.y, p2.y);

        // 判断包围盒里像素是在三角形内还是外
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                let p = IVec2::new(x, y);
                let c1 = (p1 - p0).perp_dot(p - p0);
                let c2 = (p2 - p1).perp_dot(p - p1);
                let c3 = (p0 - p2).perp_dot(p - p2);

                if (c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0) {
                    image.set(p.x as usize, p.y as usize, c);
                }

                let mut edge_flag = false;
                if c1 == 0 && Self::is_top_left_edge(p0, p1) && Self::is_in_edge(&p, p0, p1) {
                    edge_flag = true;
                } else if c2 == 0 && Self::is_top_left_edge(p1, p2) && Self::is_in_edge(&p, p1, p2)
                {
                    edge_flag = true;
                } else if c3 == 0 && Self::is_top_left_edge(p2, p0) && Self::is_in_edge(&p, p2, p0)
                {
                    edge_flag = true;
                }

                if edge_flag {
                    image.set(p.x as usize, p.y as usize, c);
                }
            }
        }
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
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(250.0, 50.0),
            &Vec2::new(50.0, 450.0),
            &Vec2::new(450.0, 450.0),
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

    #[test]
    fn test_tri_dda_flat_line() {
        // 三个点近乎共线，浮点端点
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(20.3, 100.7),
            &Vec2::new(100.0, 100.5),
            &Vec2::new(180.8, 100.2),
            &TGAColor {
                r: 50,
                g: 150,
                b: 200,
                a: 255,
            },
        );

        image
            .write_tga_file("output_triangle.tga", false, true)
            .unwrap();
    }

    #[test]
    fn test_tri_dda_steep() {
        // 瘦高三角形（陡峭边）
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });

        DrawTriangleFloat::draw(
            &mut image,
            &Vec2::new(100.0, 20.0),
            &Vec2::new(30.0, 180.0),
            &Vec2::new(170.0, 180.0),
            &TGAColor {
                r: 200,
                g: 50,
                b: 80,
                a: 255,
            },
        );

        image
            .write_tga_file("output_triangle.tga", false, true)
            .unwrap();
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
            &TGAColor {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        image
            .write_tga_file("output_triangle.tga", false, true)
            .unwrap();
    }
}
