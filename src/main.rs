mod datatype;
mod drawline;
mod drawtriangle;
mod tgaimage;

use drawtriangle::DrawTriangle;
use rand::{Rng, RngExt};
use tgaimage::{TGAColor, TGAImage, TGAImageType};

// test
use drawline::Bresenham;

use crate::datatype::Point2D;

fn main() {
    let mut rng = rand::rng();
    let mut img = TGAImage::new(200, 200, TGAImageType::RGBA);
    img.set_background_color(&TGAColor {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    });

    for _ in 0..1600 {
        let ramdom_color = TGAColor::new(
            rng.random_range(1..255),
            rng.random_range(1..255),
            rng.random_range(1..255),
            255,
        );

        let p0 = Point2D {
            x: rng.random_range(0..200),
            y: rng.random_range(0..100),
        };

        let p1 = Point2D {
            x: rng.random_range(0..200),
            y: rng.random_range(100..200),
        };

        Bresenham::draw(&mut img, &p0, &p1, &ramdom_color);
    }

    img.write_tga_file("output.tga", false, true).unwrap();
    println!("Generated output.tga");
}
