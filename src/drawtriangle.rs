use crate::datatype::Point2D;
use crate::drawline::{Drawline, Bresenham};
use crate::tgaimage::{TGAColor, TGAImage, TGAImageType};

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
