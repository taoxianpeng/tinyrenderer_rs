use crate::datatype::Point2D;
use crate::tgaimage::*;

pub trait Drawline {
    fn draw(image: &mut TGAImage, p0: &Point2D, p1: &Point2D, c: &TGAColor);
}

pub struct Bresenham;

impl Bresenham {
    pub fn draw(image: &mut TGAImage, p0: &Point2D, p1: &Point2D, c: &TGAColor) {
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

    #[test]
    fn test_line_1() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 100 },
            &Point2D { x: 400, y: 400 },
            &TGAColor {
                r: 100,
                g: 23,
                b: 30,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_2() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 200 },
            &Point2D { x: 400, y: 200 },
            &TGAColor {
                r: 100,
                g: 23,
                b: 30,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_3() {
        let mut image = TGAImage::new(500, 500, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 100, y: 300 },
            &Point2D { x: 400, y: 100 },
            &TGAColor {
                r: 100,
                g: 23,
                b: 30,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_4_reversed_x() {
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 180, y: 30 },
            &Point2D { x: 20, y: 170 },
            &TGAColor {
                r: 200,
                g: 50,
                b: 80,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_steep_positive() {
        // 陡峭正斜率：|dy| > |dx|
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 30, y: 30 },
            &Point2D { x: 80, y: 170 },
            &TGAColor {
                r: 50,
                g: 150,
                b: 200,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }

    #[test]
    fn test_line_steep_negative() {
        // 陡峭负斜率：|dy| > |dx|
        let mut image = TGAImage::new(200, 200, TGAImageType::RGB);
        image.set_background_color(&TGAColor {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        });
        Bresenham::draw(
            &mut image,
            &Point2D { x: 80, y: 170 },
            &Point2D { x: 30, y: 30 },
            &TGAColor {
                r: 50,
                g: 150,
                b: 200,
                a: 255,
            },
        );
        image.write_tga_file("output.tga", false, true).unwrap();
    }
}
