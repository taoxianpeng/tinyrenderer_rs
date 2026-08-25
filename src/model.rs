use glam::{Vec2, Vec3};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::drawline::WHITE;
use crate::renderpipeline::{Varying, VertexInput};

pub type VertIndice = [u32; 3];
pub type Face = Vec<VertIndice>;

/// 从 OBJ 文件组装顶点输入（仅处理三角形面）
pub fn load_model(path: &str, enable_tbn: bool) -> Option<Vec<VertexInput>> {
    let model: Model = Model::new(Path::new(path));
    println!(
        "模型加载成功: {} 顶点, {} 面",
        model.verts().len() - 1,
        model.faces().len() - 1,
    );

    let mut vertices: Vec<VertexInput> = Vec::new();
    for face in model.faces() {
        if face.len() == 3 {
            // 求切线空间中TBN矩阵的T：每个三角形一份，写进该三角形的 3 个顶点
            let t_vec = match enable_tbn {
                true => {
                    let pos_index_1 = face[0][0];
                    let uv_index_1 = face[0][1];
                    let p1 = model.verts()[pos_index_1 as usize];
                    let uv1 = model.texture_verts()[uv_index_1 as usize].truncate();

                    let pos_index_2 = face[1][0];
                    let uv_index_2 = face[1][1];
                    let p2 = model.verts()[pos_index_2 as usize];
                    let uv2 = model.texture_verts()[uv_index_2 as usize].truncate();

                    let pos_index_3 = face[2][0];
                    let uv_index_3 = face[2][1];
                    let p3 = model.verts()[pos_index_3 as usize];
                    let uv3 = model.texture_verts()[uv_index_3 as usize].truncate();

                    let edge1 = p2 - p1;
                    let edge2 = p3 - p1;
                    let delta_uv1 = uv2 - uv1;
                    let delta_u1 = delta_uv1.x;
                    let delta_v1 = delta_uv1.y;
                    let delta_uv2 = uv3 - uv1;
                    let delta_u2 = delta_uv2.x;
                    let delta_v2 = delta_uv2.y;

                    let f = 1.0 / (delta_u1 * delta_v2 - delta_u2 * delta_v1);
                    Vec3::new(
                        f * (delta_v2 * edge1.x - delta_v1 * edge2.x),
                        f * (delta_v2 * edge1.y - delta_v1 * edge2.y),
                        f * (delta_v2 * edge1.z - delta_v1 * edge2.z),
                    )
                    .normalize()
                    // 注：B = cross(T, N) 留到顶点着色器中计算，省一个 varyings
                }
                // 未启用 TBN 时用占位值，保持 varyings 数量一致
                false => Vec3::ZERO,
            };

            for idx in face {
                let pos = model.verts()[idx[0] as usize];
                let normal = model.vert_normals()[idx[2] as usize];
                let texcoord = {
                    let vt = model.texture_verts()[idx[1] as usize];
                    Vec2::new(vt.x, vt.y)
                };
                vertices.push(VertexInput {
                    pos,
                    varyings: vec![
                        Varying::Color(WHITE),
                        Varying::Vec3(normal),
                        Varying::Vec2(texcoord),
                        Varying::Vec3(t_vec), // varyings[3]: 切线 T (TBN)
                    ],
                });
            }
        }
    }
    println!(
        "组装 {} 个顶点输入 ({} 个三角形)",
        vertices.len(),
        vertices.len() / 3
    );
    Some(vertices)
}

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

        let mut ret = Self {
            verts: vec![Vec3::ZERO],
            texture_verts: vec![Vec3::ZERO],
            vert_normals: vec![Vec3::ZERO],
            faces: vec![vec![[0, 0, 0]]],
        };

        for line in reader.lines() {
            let line = line.unwrap();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((prefix, _)) = line.split_once(' ') else {
                continue;
            };
            match prefix {
                "v" => {
                    let vert = Self::parse_vertex(&line, "v").unwrap();
                    ret.verts.push(vert);
                }
                "vt" => {
                    let tex_indice = Self::parse_vertex(&line, "vt").unwrap();
                    ret.texture_verts.push(tex_indice);
                }
                "vn" => {
                    let normal = Self::parse_vertex(&line, "vn").unwrap();
                    ret.vert_normals.push(normal);
                }
                "f" => {
                    let face = Self::parse_face_vert_indice(&line).unwrap();
                    ret.faces.push(face.to_vec());
                }

                _ => {}
            }
        }

        ret
    }

    fn parse_vertex(line: &str, prefix: &str) -> Result<Vec3, String> {
        if line.starts_with(prefix) {
            let verts = line
                .strip_prefix(prefix)
                .ok_or(format!("missing '{prefix}' prefix"))?
                .trim()
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

    fn parse_face_vert_indice(line: &str) -> Result<Vec<VertIndice>, String> {
        let prefix = "f ";
        if line.starts_with(prefix) {
            let indices: Vec<VertIndice> = line
                .strip_prefix(prefix)
                .ok_or(format!("missing '{prefix}' prefix"))?
                .trim()
                .split_whitespace()
                .map(|token| -> Result<VertIndice, String> {
                    let parts: Vec<&str> = token.split('/').collect();
                    let t0 = parts[0]
                        .parse::<u32>()
                        .map_err(|e| format!("parse vertex index: {e}"))?;
                    let t1 = parts
                        .get(1)
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    let t2 = parts
                        .get(2)
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    Ok([t0, t1, t2])
                })
                .collect::<Result<Vec<_>, _>>()?;

            return Ok(indices);
        }

        Err("invalid face data".to_string())
    }

    // ---- 只读访问器 ----

    pub fn verts(&self) -> &Vec<Vec3> {
        &self.verts
    }

    pub fn texture_verts(&self) -> &Vec<Vec3> {
        &self.texture_verts
    }

    pub fn vert_normals(&self) -> &Vec<Vec3> {
        &self.vert_normals
    }

    pub fn faces(&self) -> &Vec<Face> {
        &self.faces
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_vertex() {
        let test = "v 0.11526 -0.700717 0.0677257";
        let ret = Model::parse_vertex(test, "v").unwrap();
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

    #[test]
    fn test_parse_face() {
        let test = "f 644/2833/644 653/674/653 673/673/673";
        let ret = Model::parse_face_vert_indice(test).unwrap();
        for (i, vert) in ret.iter().enumerate() {
            println!(
                "  vertex {}: v={}, vt={}, vn={}",
                i, vert[0], vert[1], vert[2]
            );
        }

        assert_eq!(
            ret,
            vec![[644, 2833, 644], [653, 674, 653], [673, 673, 673],]
        );
    }

    #[test]
    fn test_model_load() {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("assert/diablo3_pose/diablo3_pose.obj");

        let model = Model::new(&path);

        // OBJ 索引从 1 开始，new() 中插入了占位的 [0,0,0]，所以实际数量会多 1
        assert_eq!(model.verts.len(), 2520); // 2519 + 1 占位
        assert_eq!(model.texture_verts.len(), 3264); // 3263 + 1 占位
        assert_eq!(model.vert_normals.len(), 2520); // 2519 + 1 占位
        assert_eq!(model.faces.len(), 5023); // 5022 + 1 占位

        // 验证第一个真实面（索引 0 是占位，索引 1 是第一个 OBJ 面）
        // OBJ: "f 6/1/6 5/2/5 8/3/8"
        let first_face = &model.faces[1];
        assert_eq!(first_face.len(), 3);
        assert_eq!(first_face[0], [6, 1, 6]);
        assert_eq!(first_face[1], [5, 2, 5]);
        assert_eq!(first_face[2], [8, 3, 8]);

        println!(
            "模型加载成功: {} 顶点, {} 纹理, {} 法线, {} 面",
            model.verts.len(),
            model.texture_verts.len(),
            model.vert_normals.len(),
            model.faces.len(),
        );
    }
}
