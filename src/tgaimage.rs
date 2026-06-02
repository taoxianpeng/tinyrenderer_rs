use std::arch::x86_64::_bittest;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAX_CHUNK_LENGTH: u8 = 128;

#[derive(Debug)]
pub enum TgaError {
    Io(io::Error),
    BadHeader(&'static str),
    UnsupportedFormat(u8),
    BadData(&'static str),
}

pub enum TGAImageType {
    Grayscale = 1,
    RGB = 3,
    RGBA = 4,
}


impl fmt::Display for TgaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TgaError::Io(e) => write!(f, "I/O error: {}", e),
            TgaError::BadHeader(msg) => write!(f, "Invalid TGA header: {}", msg),
            TgaError::UnsupportedFormat(code) => {
                write!(f, "Unsupported TGA format (data type code {}). Only 2, 3, 10, 11 are supported.", code)
            }
            TgaError::BadData(msg) => write!(f, "Corrupt TGA data: {}", msg),
        }
    }
}

impl From<io::Error> for TgaError {
    fn from(e: io::Error) -> Self {
        TgaError::Io(e)
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct TgaHeader {
    id_length: u8,
    color_map_type: u8,
    data_type_code: u8,
    color_map_origin: u16,
    color_map_length: u16,
    color_map_depth: u8,
    x_origin: u16,
    y_origin: u16,
    width: u16,
    height: u16,
    bits_per_pixel: u8,
    image_descriptor: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TGAColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl TGAColor {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        TGAColor { r, g, b, a }
    }
}

impl Default for TGAColor {
    fn default() -> Self {
        TGAColor { r: 0, g: 0, b: 0, a: 0 }
    }
}

#[derive(Debug)]
pub struct TGAImage {
    w: usize,
    h: usize,
    bpp: u8,
    data: Vec<u8>,
}

impl TGAImage {
    pub fn new(w: usize, h: usize, bpp: TGAImageType) -> Self {
        let _bpp = match bpp {
            TGAImageType::Grayscale => 1,
            TGAImageType::RGB => 3,
            TGAImageType::RGBA => 4,
        };

        TGAImage {
            w,
            h,
            bpp: _bpp,
            data: vec![0; w * h * (_bpp as usize)],
        }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn bytes_per_pixel(&self) -> u8 {
        self.bpp
    }

    pub fn get(&self, x: usize, y: usize) -> Option<TGAColor> {
        if self.data.is_empty() || x >= self.w || y >= self.h {
            return None;
        }
        let idx = (x + y * self.w) * self.bpp as usize;
        let mut c = TGAColor::default();
        if self.bpp > 0 {
            c.b = self.data[idx];
        }
        if self.bpp > 1 {
            c.g = self.data[idx + 1];
        }
        if self.bpp > 2 {
            c.r = self.data[idx + 2];
        }
        if self.bpp > 3 {
            c.a = self.data[idx + 3];
        }
        Some(c)
    }

    pub fn set(&mut self, x: usize, y: usize, c: TGAColor) {
        if self.data.is_empty() || x >= self.w || y >= self.h {
            return;
        }
        let idx = (x + y * self.w) * self.bpp as usize;
        if self.bpp > 0 {
            self.data[idx] = c.b;
        }
        if self.bpp > 1 {
            self.data[idx + 1] = c.g;
        }
        if self.bpp > 2 {
            self.data[idx + 2] = c.r;
        }
        if self.bpp > 3 {
            self.data[idx + 3] = c.a;
        }
    }

    pub fn read_tga_file<P: AsRef<Path>>(&mut self, filename: P) -> Result<(), TgaError> {
        let mut file = File::open(filename.as_ref())?;

        let mut header_bytes = [0u8; 18];
        file.read_exact(&mut header_bytes)?;
        let header: TgaHeader = unsafe { std::ptr::read(header_bytes.as_ptr() as *const TgaHeader) };

        if header.data_type_code != 2
            && header.data_type_code != 3
            && header.data_type_code != 10
            && header.data_type_code != 11
        {
            return Err(TgaError::UnsupportedFormat(header.data_type_code));
        }

        self.w = header.width as usize;
        self.h = header.height as usize;
        self.bpp = header.bits_per_pixel / 8;

        if self.w == 0 || self.h == 0 || (self.bpp != 1 && self.bpp != 3 && self.bpp != 4) {
            return Err(TgaError::BadHeader("bad bpp or width/height value"));
        }

        let nbytes = self.bpp as usize * self.w * self.h;
        self.data.resize(nbytes, 0);

        // Skip image ID and color map data to reach pixel data
        let color_map_size = header.color_map_type as u32
            * header.color_map_length as u32
            * (header.color_map_depth as u32 / 8);
        let pixel_offset = 18 + header.id_length as u64 + color_map_size as u64;
        file.seek(SeekFrom::Start(pixel_offset))?;

        match header.data_type_code {
            2 | 3 => file.read_exact(&mut self.data)?,
            10 | 11 => self.load_rle_data(&mut file)?,
            _ => unreachable!(),
        }

        if header.image_descriptor & 0x20 == 0 {
            self.flip_vertically();
        }
        if header.image_descriptor & 0x10 != 0 {
            self.flip_horizontally();
        }

        Ok(())
    }

    fn load_rle_data(&mut self, file: &mut File) -> Result<(), TgaError> {
        let pixel_count = self.w * self.h;
        let mut current_pixel = 0usize;
        let mut current_byte = 0usize;
        let mut color_buffer = [0u8; 4];

        while current_pixel < pixel_count {
            let mut chunk_header = [0u8; 1];
            file.read_exact(&mut chunk_header)?;
            let chunk_header = chunk_header[0];

            if chunk_header < 128 {
                let count = (chunk_header + 1) as usize;
                for _ in 0..count {
                    file.read_exact(&mut color_buffer[..self.bpp as usize])?;
                    for t in 0..self.bpp as usize {
                        self.data[current_byte] = color_buffer[t];
                        current_byte += 1;
                    }
                    current_pixel += 1;
                    if current_pixel > pixel_count {
                        return Err(TgaError::BadData("too many pixels read"));
                    }
                }
            } else {
                let count = (chunk_header - 127) as usize;
                file.read_exact(&mut color_buffer[..self.bpp as usize])?;
                for _ in 0..count {
                    for t in 0..self.bpp as usize {
                        self.data[current_byte] = color_buffer[t];
                        current_byte += 1;
                    }
                    current_pixel += 1;
                    if current_pixel > pixel_count {
                        return Err(TgaError::BadData("too many pixels read"));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn write_tga_file<P: AsRef<Path>>(
        &self,
        filename: P,
        vflip: bool,
        rle: bool,
    ) -> Result<(), TgaError> {
        if self.bpp != 1 && self.bpp != 3 && self.bpp != 4 {
            return Err(TgaError::BadHeader(
                "invalid bytes per pixel: must be 1 (grayscale), 3 (RGB), or 4 (RGBA)",
            ));
        }
        let mut file = File::create(filename.as_ref())?;

        let header = TgaHeader {
            bits_per_pixel: self.bpp * 8,
            width: self.w as u16,
            height: self.h as u16,
            data_type_code: if self.bpp == 1 {
                if rle { 11 } else { 3 }
            } else {
                if rle { 10 } else { 2 }
            },
            image_descriptor: {
                let mut desc = if vflip { 0x00 } else { 0x20 };
                if self.bpp == 4 {
                    desc |= 8; // 8-bit alpha channel depth
                }
                desc
            },
            ..Default::default()
        };

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                (&header as *const TgaHeader) as *const u8,
                std::mem::size_of::<TgaHeader>(),
            )
        };
        file.write_all(header_bytes)?;

        if rle {
            self.unload_rle_data(&mut file)?;
        } else {
            file.write_all(&self.data)?;
        }

        let developer_area_ref = [0u8; 4];
        let extension_area_ref = [0u8; 4];
        let footer = b"TRUEVISION-XFILE.\0";
        file.write_all(&developer_area_ref)?;
        file.write_all(&extension_area_ref)?;
        file.write_all(footer)?;

        Ok(())
    }

    fn unload_rle_data(&self, file: &mut File) -> Result<(), TgaError> {
        let npixels = self.w * self.h;
        let mut curpix = 0usize;

        while curpix < npixels {
            let chunk_start = curpix * self.bpp as usize;
            let mut curbyte = curpix * self.bpp as usize;
            let mut run_length: u8 = 1;
            let mut raw = true;

            while (curpix + run_length as usize) < npixels && run_length < MAX_CHUNK_LENGTH {
                let mut succ_eq = true;
                for t in 0..self.bpp as usize {
                    if self.data[curbyte + t] != self.data[curbyte + t + self.bpp as usize] {
                        succ_eq = false;
                        break;
                    }
                }
                curbyte += self.bpp as usize;

                if run_length == 1 {
                    raw = !succ_eq;
                }
                if raw && succ_eq {
                    run_length -= 1;
                    break;
                }
                if !raw && !succ_eq {
                    break;
                }
                run_length += 1;
            }

            curpix += run_length as usize;

            if raw {
                file.write_all(&[run_length - 1])?;
                file.write_all(
                    &self.data[chunk_start..chunk_start + run_length as usize * self.bpp as usize],
                )?;
            } else {
                file.write_all(&[run_length + 127])?;
                file.write_all(&self.data[chunk_start..chunk_start + self.bpp as usize])?;
            }
        }

        Ok(())
    }

    pub fn flip_horizontally(&mut self) {
        let bpp = self.bpp as usize;
        for i in 0..self.w / 2 {
            for j in 0..self.h {
                for b in 0..bpp {
                    let left = (i + j * self.w) * bpp + b;
                    let right = (self.w - 1 - i + j * self.w) * bpp + b;
                    self.data.swap(left, right);
                }
            }
        }
    }

    pub fn flip_vertically(&mut self) {
        let bpp = self.bpp as usize;
        for i in 0..self.w {
            for j in 0..self.h / 2 {
                for b in 0..bpp {
                    let top = (i + j * self.w) * bpp + b;
                    let bottom = (i + (self.h - 1 - j) * self.w) * bpp + b;
                    self.data.swap(top, bottom);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn test_path() -> std::path::PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("tinyrenderer_test_{}.tga", n))
    }

    fn checker_image(w: usize, h: usize, bpp: u8) -> TGAImage {
        let mut img = TGAImage::new(w, h, bpp);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 64) % 256) as u8;
                let g = ((y * 64) % 256) as u8;
                let b = 128u8;
                let a = 255u8;
                img.set(x, y, TGAColor::new(r, g, b, a));
            }
        }
        img
    }

    #[test]
    fn test_create_and_get_pixel() {
        let img = checker_image(4, 4, 4);
        let c = img.get(1, 1).unwrap();
        assert_eq!(c.r, 64);
        assert_eq!(c.g, 64);
        assert_eq!(c.b, 128);
        assert_eq!(c.a, 255);

        let c0 = img.get(0, 0).unwrap();
        assert_eq!(c0, TGAColor::new(0, 0, 128, 255));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let img = checker_image(2, 2, 4);
        assert!(img.get(2, 2).is_none());
        assert!(img.get(5, 0).is_none());
        assert!(img.get(0, 5).is_none());
    }

    #[test]
    fn test_set_out_of_bounds_no_panic() {
        let mut img = checker_image(2, 2, 4);
        img.set(10, 10, TGAColor::new(255, 0, 0, 255));
        assert!(img.get(10, 10).is_none());
    }

    #[test]
    fn test_width_height_bpp() {
        let img = TGAImage::new(10, 20, 3);
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 20);
        assert_eq!(img.bytes_per_pixel(), 3);
    }

    #[test]
    fn test_flip_horizontally() {
        let mut img = checker_image(4, 4, 4);
        let tl = img.get(0, 0).unwrap();
        let tr = img.get(3, 0).unwrap();

        img.flip_horizontally();

        assert_eq!(img.get(3, 0).unwrap(), tl);
        assert_eq!(img.get(0, 0).unwrap(), tr);
    }

    #[test]
    fn test_flip_vertically() {
        let mut img = checker_image(4, 4, 4);
        let tl = img.get(0, 0).unwrap();
        let bl = img.get(0, 3).unwrap();

        img.flip_vertically();

        assert_eq!(img.get(0, 3).unwrap(), tl);
        assert_eq!(img.get(0, 0).unwrap(), bl);
    }

    #[test]
    fn test_double_flip_is_identity() {
        let mut img = checker_image(5, 5, 4);
        let original = pixels(&img, 5, 5);

        img.flip_horizontally();
        img.flip_horizontally();
        assert_eq!(pixels(&img, 5, 5), original);

        img.flip_vertically();
        img.flip_vertically();
        assert_eq!(pixels(&img, 5, 5), original);
    }

    fn pixels(img: &TGAImage, w: usize, h: usize) -> Vec<TGAColor> {
        let mut result = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                result.push(img.get(x, y).unwrap());
            }
        }
        result
    }

    #[test]
    fn test_grayscale_roundtrip() {
        let path = test_path();
        let mut img = TGAImage::new(3, 3, 1);
        img.set(0, 0, TGAColor::new(0, 0, 0, 0));
        img.set(1, 1, TGAColor::new(0, 0, 128, 0));
        img.set(2, 2, TGAColor::new(0, 0, 255, 0));

        img.write_tga_file(&path, false, false).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), 3);
        assert_eq!(loaded.height(), 3);
        assert_eq!(loaded.bytes_per_pixel(), 1);
        assert_eq!(loaded.get(1, 1).unwrap().b, 128);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_rgb_roundtrip() {
        let path = test_path();
        let img = checker_image(4, 4, 3);

        img.write_tga_file(&path, false, false).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), 4);
        assert_eq!(loaded.height(), 4);
        assert_eq!(loaded.bytes_per_pixel(), 3);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.get(x, y).unwrap(), loaded.get(x, y).unwrap());
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_rgba_roundtrip() {
        let path = test_path();
        let img = checker_image(4, 4, 4);

        img.write_tga_file(&path, false, false).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), 4);
        assert_eq!(loaded.height(), 4);
        assert_eq!(loaded.bytes_per_pixel(), 4);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.get(x, y).unwrap(), loaded.get(x, y).unwrap());
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_rle_roundtrip() {
        let path = test_path();
        let img = checker_image(4, 4, 4);

        img.write_tga_file(&path, false, true).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), 4);
        assert_eq!(loaded.height(), 4);
        assert_eq!(loaded.bytes_per_pixel(), 4);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.get(x, y).unwrap(), loaded.get(x, y).unwrap());
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_rle_uniform_image() {
        let path = test_path();
        let mut img = TGAImage::new(8, 8, 4);
        let red = TGAColor::new(255, 0, 0, 255);
        for y in 0..8 {
            for x in 0..8 {
                img.set(x, y, red);
            }
        }

        img.write_tga_file(&path, false, true).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), 8);
        assert_eq!(loaded.height(), 8);

        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(loaded.get(x, y).unwrap(), red);
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = TGAImage::new(1, 1, 4).read_tga_file("/nonexistent/path/file.tga");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_format() {
        // Create a minimal TGA file with unsupported data type code
        let path = test_path();
        let mut file = File::create(&path).unwrap();
        let mut header = [0u8; 18];
        header[2] = 1; // unsupported data type code 1 (color-mapped)
        file.write_all(&header).unwrap();
        drop(file);

        let mut img = TGAImage::new(1, 1, 4);
        let result = img.read_tga_file(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_empty_image() {
        let img = TGAImage::new(0, 0, 4);
        assert!(img.get(0, 0).is_none());
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
    }

    #[test]
    fn test_set_modifies_pixel() {
        let mut img = TGAImage::new(2, 2, 4);
        let blue = TGAColor::new(0, 0, 255, 255);
        img.set(1, 0, blue);
        assert_eq!(img.get(1, 0).unwrap(), blue);
        assert_ne!(img.get(0, 0).unwrap(), blue);
    }

    #[test]
    fn test_default_color_is_black() {
        let c = TGAColor::default();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 0);
    }

    #[test]
    fn test_large_image_rle_roundtrip() {
        let path = test_path();
        let w = 64;
        let h = 64;
        let mut img = TGAImage::new(w, h, 4);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 4) as u8;
                let g = (y * 4) as u8;
                let b = ((x + y) * 2) as u8;
                let a = 255;
                img.set(x, y, TGAColor::new(r, g, b, a));
            }
        }

        img.write_tga_file(&path, false, true).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();
        assert_eq!(loaded.width(), w);
        assert_eq!(loaded.height(), h);

        for y in 0..h {
            for x in 0..w {
                assert_eq!(img.get(x, y).unwrap(), loaded.get(x, y).unwrap());
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_vflip_false_roundtrip() {
        let path = test_path();
        let img = checker_image(4, 4, 4);

        img.write_tga_file(&path, false, false).unwrap();

        let mut loaded = TGAImage::new(1, 1, 1);
        loaded.read_tga_file(&path).unwrap();

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.get(x, y).unwrap(), loaded.get(x, y).unwrap());
            }
        }

        std::fs::remove_file(&path).ok();
    }
}
