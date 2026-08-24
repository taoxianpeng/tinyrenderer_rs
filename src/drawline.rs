pub type Point2D = glam::IVec2;
pub use crate::tgaimage::*;
pub use glam::{IVec2, Vec2};


/// DDA 浮点直线绘制 — 接受浮点坐标端点
pub struct DDA;

impl DDA {
    pub fn draw(image: &mut TGAImage, p0: &Vec2, p1: &Vec2, c: &TGAColor) {
        let mut x0 = p0.x;
        let mut y0 = p0.y;
        let mut x1 = p1.x;
        let mut y1 = p1.y;

        // 陡峭判断：|dy| > |dx| 则交换 x/y，保证沿长轴步进
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

        // 主轴方向的像素步数
        let steps = dx.round().abs() as usize;
        if steps == 0 {
            // 退化为单个像素
            let px = x0.round() as usize;
            let py = y0.round() as usize;
            if steep {
                image.set(py, px, c);
            } else {
                image.set(px, py, c);
            }
            return;
        }

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = x0 + t * dx;
            let y = y0 + t * dy;

            let px = x.round() as usize;
            let py = y.round() as usize;

            if steep {
                image.set(py, px, c);
            } else {
                image.set(px, py, c);
            }
        }
    }
}

pub struct Bresenham;

impl Bresenham {
    pub fn draw(image: &mut TGAImage, p0: &IVec2, p1: &IVec2, c: &TGAColor) {
        let mut x0 = p0.x;
        let mut y0 = p0.y;
        let mut x1 = p1.x;
        let mut y1 = p1.y;

        // 若 |dy| > |dx|（陡峭），交换 x 和 y，以长方向为主循环
        let steep = (y1 - y0).abs() > (x1 - x0).abs();
        if steep {
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
        }

        // 确保主循环变量单调递增
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
                // 交换过坐标，画点时换回：set(y_alg, x_alg)
                image.set(y as usize, x as usize, c);
            } else {
                image.set(x as usize, y as usize, c);
            }

            // if d >= 0 {
            // y += y_step;
            // d += 2 * (dy_abs - dx);
            // } else {
            // d += 2 * dy_abs;
            // }
            let k = if d >= 0 { 1 } else { 0 };
            y += y_step * k;
            d += 2 * (dy_abs - dx * k);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // 测试颜色（f32，0.0–1.0 范围）
    const DARK_RED: TGAColor = TGAColor::new(100.0 / 255.0, 23.0 / 255.0, 30.0 / 255.0, 1.0);
    const PINK: TGAColor = TGAColor::new(200.0 / 255.0, 50.0 / 255.0, 80.0 / 255.0, 1.0);
    const SKY_BLUE: TGAColor = TGAColor::new(50.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0, 1.0);

    #[test]
    fn test_line_1() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 100 },
            &Point2D { x: 400, y: 400 },
            &DARK_RED,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_2() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 200 },
            &Point2D { x: 400, y: 200 },
            &DARK_RED,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_3() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 300 },
            &Point2D { x: 400, y: 100 },
            &DARK_RED,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_4_reversed_x() {
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 180, y: 30 },
            &Point2D { x: 20, y: 170 },
            &PINK,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_steep_positive() {
        // 陡峭正斜率：|dy| > |dx|
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 30, y: 30 },
            &Point2D { x: 80, y: 170 },
            &SKY_BLUE,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_steep_negative() {
        // 陡峭负斜率：|dy| > |dx|
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        Bresenham::draw(
            &mut image,
            &Point2D { x: 80, y: 170 },
            &Point2D { x: 30, y: 30 },
            &SKY_BLUE,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    // ======== DDA 浮点版本测试 ========

    #[test]
    fn test_dda_basic_diagonal() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        DDA::draw(
            &mut image,
            &Vec2::new(100.0, 100.0),
            &Vec2::new(400.0, 400.0),
            &DARK_RED,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_dda_float_endpoints() {
        // 浮点数端点，非整数位置
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        DDA::draw(
            &mut image,
            &Vec2::new(30.3, 50.7),
            &Vec2::new(170.8, 130.2),
            &PINK,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_dda_steep() {
        // 陡峭线
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&WHITE);
        DDA::draw(
            &mut image,
            &Vec2::new(30.0, 30.0),
            &Vec2::new(80.0, 170.0),
            &SKY_BLUE,
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_dda_same_point() {
        // 同一个点
        let mut image = TGAImage::new(10, 10, TGAImageType::RGB);
        DDA::draw(
            &mut image,
            &Vec2::new(5.3, 5.7),
            &Vec2::new(5.3, 5.7),
            &RED,
        );
    }
}
