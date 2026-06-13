use std::fs::{self, File};
use std::io::BufReader;
use std::mem::zeroed;
use std::path::Path;

use crate::datatype::Vec3;

pub type VertIndice = [u32; 3];
pub type Face = [VertIndice; 3];

pub struct Model {
    verts: Vec<Vec3>,
    texture_verts: Vec<Vec3>,
    vert_normals: Vec<Vec3>,
    faces: Vec<Face>,
}

impl Model {
    pub fn new(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        Self {
            verts: vec![Vec3::new()],
            texture_verts: vec![Vec3::new()],
            vert_normals: vec![Vec3::new()],
            faces: vec![[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        }
    }

    fn parse_vertex(line: &str) -> Result<Vec3, String> {
        if line.starts_with("v ") {
            let verts = line
                .strip_prefix("v ")
                .ok_or("missing 'v ' prefix")?
                .splitn(3, ' ')
                .map(|s| s.parse::<f32>().map_err(|e| format!("parse error: {e}")))
                .collect::<Result<Vec<_>, _>>()?;

            return Ok(Vec3 {
                x: verts[0],
                y: verts[1],
                z: verts[2],
            });
        }

        Err("invalid vertex data".to_string())
    }

    fn parse_texture_vertex() {}

    fn parse_face_vert_indice() {}
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_vertex() {
        let test = "v 0.11526 -0.700717 0.0677257";
        let ret = Model::parse_vertex(test).unwrap();
        println!("ret: {}", ret);
        assert_eq!(
            ret,
            Vec3 {
                x: 0.11526,
                y: -0.700717,
                z: 0.0677257
            }
        );
    }
}
