mod tgaimage;

use tgaimage::{TGAColor, TGAImage, TGAImageType};

fn main() {
    let red = TGAColor::new(255, 0, 0, 255);
    let green = TGAColor::new(0, 255, 0, 255);
    let blue = TGAColor::new(0, 0, 255, 255);
    let white = TGAColor::new(255, 255, 255, 255);

    let mut img = TGAImage::new(200, 200, TGAImageType::RGBA);
    for y in 0..200 {
        for x in 0..200 {
            if x < 100 && y < 100 {
                img.set(x, y, red);
            } else if x >= 100 && y < 100 {
                img.set(x, y, green);
            } else if x < 100 && y >= 100 {
                img.set(x, y, blue);
            } else {
                img.set(x, y, white);
            }
        }
    }
    img.write_tga_file("output.tga", false, true).unwrap();
    println!("Generated output.tga");
}
