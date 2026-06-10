use crate::datatype::Point2D;
use crate::tgaimage::*;

trait Drawline {
    fn draw(image: &mut TGAImage, p0: &Point2D, p1: &Point2D, c: &TGAColor);
}

pub struct Bresenham<'a> {
    image: &'a mut TGAImage,
    p0: Point2D,
    p1: Point2D,
    bg_color: TGAColor,
}

impl<'a> Bresenham<'a> {
    pub fn new(image: &'a mut TGAImage) -> Self {
        Bresenham {
            image,
            p0: Point2D { x: 0, y: 0 },
            p1: Point2D { x: 0, y: 0 },
            bg_color: TGAColor {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
        }
    }

    pub fn from(mut self, p: Point2D) -> Self {
        self.p0 = p;
        self
    }

    pub fn to(mut self, p: Point2D) -> Self {
        self.p1 = p;
        self
    }

    pub fn set_bg_color(mut self, c: TGAColor) -> Self {
        for x in 0..self.image.width() {
            for y in 0..self.image.height() {
                self.image.set(x, y, &c);
            }
        }
        self
    }
}

impl<'a> Drawline for Bresenham<'a> {
    fn draw(image: &mut TGAImage, p0: &Point2D, p1: &Point2D, c: &TGAColor) {
        // 确保 x0 ≤ x1，使 x 方向单调递增
        let (mut x0, mut y0, x1, y1) = if p0.x > p1.x {
            (p1.x, p1.y, p0.x, p0.y)
        } else {
            (p0.x, p0.y, p1.x, p1.y)
        };

        let dx = x1 - x0; // dx > 0 （|k| < 1 保证 dx > dy ≥ 0）
        let dy = y1 - y0; // 可正、负或零
        let dy_abs = dy.abs();

        let y_step: i32 = if y1 >= y0 { 1 } else { -1 };
        let mut d = 2 * dy_abs - dx;
        let mut y = y0;

        for x in x0..=x1 {
            image.set(x as usize, y as usize, &c);
            if d >= 0 {
                y += y_step; // 沿 y_step 方向走一步
                d += 2 * (dy_abs - dx);
            } else {
                d += 2 * dy_abs;
            }
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
}
