use std::{fmt, vec};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

impl Point2D {
    pub fn add(&self, other: &Point2D) -> Point2D {
        Point2D {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn add_mut(&mut self, other: &Point2D) {
        self.x += other.x;
        self.y += other.y;
    }

    pub fn sub(&self, other: &Point2D) -> Point2D {
        Point2D {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn sub_mut(&mut self, other: &Point2D) {
        self.x -= other.x;
        self.y -= other.y;
    }

    pub fn mul(&self, scalar: i32) -> Point2D {
        Point2D {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }

    pub fn mul_mut(&mut self, scalar: i32) {
        self.x *= scalar;
        self.y *= scalar;
    }

    pub fn div(&self, scalar: i32) -> Point2D {
        Point2D {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }

    pub fn div_mut(&mut self, scalar: i32) {
        self.x /= scalar;
        self.y /= scalar;
    }

    pub fn dist(&self, other: &Point2D) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Vec3 {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn add(&mut self, v: &Vec3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }

    pub fn sub(&mut self, v: &Vec3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct Mat4<T> {
    data: Vec<Vec<T>>,
}

impl<T> Mat4<T> 
where 
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
{
    pub fn new() -> Self {
        Mat4 {
            data: vec![vec![T::default(); 4]; 4],
        }
    }

    pub fn at(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.data.len() && col < self.data[row].len() {
            Some(&self.data[row][col])
        } else {
            None
        }
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.data.len() && col < self.data[row].len() {
            Some(&mut self.data[row][col])
        } else {
            None
        }
    }

    pub fn mul(&self, other: &Mat4<T>) -> Result<Mat4<T>, &str> {
        let rows = self.data.len();
        let cols = other.data.first().map(|r| r.len()).unwrap_or(0);
        let inner = self.data.first().map(|r| r.len()).unwrap_or(0);

        if rows == 0 || cols == 0 || inner == 0 {
            return Err("empty matrix");
        }
        if self.data.iter().any(|r| r.len() != inner) || other.data.iter().any(|r| r.len() != cols) {
            return Err("jagged matrix");
        }
        if other.data.len() != inner {
            return Err("dimension mismatch");
        }

        let mut result = vec![vec![T::default(); cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::default();
                for k in 0..inner {
                    sum = sum + self.data[i][k] * other.data[k][j];
                }
                result[i][j] = sum;
            }
        }

        Ok(Mat4 { data: result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add_1() {
        let point1 = Point2D { x: 10, y: 30 };
        let point2 = Point2D { x: 20, y: 40 };
        let new_point = point1.add(&point2);
        assert_eq!(new_point, Point2D { x: 30, y: 70 });
    }

    #[test]
    fn test_add_mut() {
        let mut point1 = Point2D { x: 10, y: 30 };
        let point2 = Point2D { x: 20, y: 40 };
        point1.add_mut(&point2);
        assert_eq!(point1, Point2D { x: 30, y: 70 });
    }

    #[test]
    fn test_sub() {
        let point1 = Point2D { x: 10, y: 30 };
        let point2 = Point2D { x: 20, y: 40 };
        let new_point = point1.sub(&point2);
        assert_eq!(new_point, Point2D { x: -10, y: -10 });
    }

    #[test]
    fn test_sub_mut() {
        let mut point1 = Point2D { x: 10, y: 30 };
        let point2 = Point2D { x: 20, y: 40 };
        point1.sub_mut(&point2);
        assert_eq!(point1, Point2D { x: -10, y: -10 });
    }

    #[test]
    fn test_mul() {
        let point = Point2D { x: 10, y: 30 };
        let new_point = point.mul(2);
        assert_eq!(new_point, Point2D { x: 20, y: 60 });
    }

    #[test]
    fn test_mul_mut() {
        let mut point = Point2D { x: 10, y: 30 };
        point.mul_mut(2);
        assert_eq!(point, Point2D { x: 20, y: 60 });
    }

    #[test]
    fn test_div() {
        let point = Point2D { x: 10, y: 30 };
        let new_point = point.div(2);
        assert_eq!(new_point, Point2D { x: 5, y: 15 });
    }

    #[test]
    fn test_div_mut() {
        let mut point = Point2D { x: 10, y: 30 };
        point.div_mut(2);
        assert_eq!(point, Point2D { x: 5, y: 15 });
    }

    #[test]
    fn test_dist() {
        let point1 = Point2D { x: 0, y: 0 };
        let point2 = Point2D { x: 3, y: 4 };
        let distance = point1.dist(&point2);
        assert_eq!(distance, 5.0);
    }

    // ─── Mat4 tests ────────────────────────────────────────────

    /// Helper: create a Mat4 from a 2D array literal
    fn mat4_from<const R: usize, const C: usize>(data: [[i32; C]; R]) -> Mat4<i32> {
        Mat4 {
            data: data.iter().map(|row| row.to_vec()).collect(),
        }
    }

    #[test]
    fn test_mat4_new_creates_4x4_zeros() {
        let m: Mat4<i32> = Mat4::new();
        assert_eq!(m.data.len(), 4);
        for row in &m.data {
            assert_eq!(row.len(), 4);
            assert!(row.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn test_mat4_at_out_of_bounds() {
        let m = mat4_from([[1, 2], [3, 4]]);
        // row out of range
        assert_eq!(m.at(2, 0), None);
        // col out of range
        assert_eq!(m.at(0, 2), None);
        // both out of range
        assert_eq!(m.at(9, 9), None);
    }

    #[test]
    fn test_mat4_at_valid() {
        let m = mat4_from([[1, 2], [3, 4]]);
        assert_eq!(m.at(0, 0), Some(&1));
        assert_eq!(m.at(0, 1), Some(&2));
        assert_eq!(m.at(1, 0), Some(&3));
        assert_eq!(m.at(1, 1), Some(&4));
    }

    #[test]
    fn test_mat4_at_mut_modify() {
        let mut m = mat4_from([[1, 2], [3, 4]]);
        if let Some(v) = m.at_mut(0, 1) {
            *v = 99;
        }
        assert_eq!(m.at(0, 1), Some(&99));
        // modifying out-of-bounds is a no-op
        assert_eq!(m.at_mut(5, 0), None);
    }

    #[test]
    fn test_mat4_mul_basic() {
        //   [1 2]   [5 6]    [1*5+2*7  1*6+2*8]   [19 22]
        //   [3 4] × [7 8] =  [3*5+4*7  3*6+4*8] = [43 50]
        let a = mat4_from([[1, 2], [3, 4]]);
        let b = mat4_from([[5, 6], [7, 8]]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.data[0], vec![19, 22]);
        assert_eq!(result.data[1], vec![43, 50]);
    }

    #[test]
    fn test_mat4_mul_rectangular() {
        // 2×3 × 3×2 = 2×2
        // [1 2 3]   [7 10]    [1*7+2*8+3*9   1*10+2*11+3*12]   [50  68]
        // [4 5 6] × [8 11] =  [4*7+5*8+6*9   4*10+5*11+6*12] = [122 167]
        //           [9 12]
        let a = mat4_from([[1, 2, 3], [4, 5, 6]]);
        let b = mat4_from([[7, 10], [8, 11], [9, 12]]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.data[0], vec![50, 68]);
        assert_eq!(result.data[1], vec![122, 167]);
    }

    #[test]
    fn test_mat4_mul_identity_like() {
        // [1 0]   [a b]   [a b]
        // [0 1] × [c d] = [c d]
        let identity = mat4_from([[1, 0], [0, 1]]);
        let m = mat4_from([[7, 8], [9, 10]]);
        let result = identity.mul(&m).unwrap();
        assert_eq!(result.data, vec![vec![7, 8], vec![9, 10]]);
    }

    #[test]
    fn test_mat4_mul_empty() {
        let a = Mat4::<i32> { data: Vec::new() };
        let b = mat4_from([[1, 2], [3, 4]]);
        assert_eq!(a.mul(&b), Err("empty matrix"));
        assert_eq!(b.mul(&a), Err("empty matrix"));
    }

    #[test]
    fn test_mat4_mul_zero_matrix() {
        // 0 矩阵乘以任何矩阵 = 0 矩阵
        let zero = Mat4::new(); // 4×4 零矩阵
        let m = mat4_from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let result = zero.mul(&m).unwrap();
        assert_eq!(result.data, vec![vec![0; 4]; 4]);

        // 反过来也是 0 矩阵
        let result2 = m.mul(&zero).unwrap();
        assert_eq!(result2.data, vec![vec![0; 4]; 4]);
    }

    #[test]
    fn test_mat4_mul_dimension_mismatch() {
        // 2×2 × 3×3 — inner dimension mismatch (2 ≠ 3)
        let a = mat4_from([[1, 2], [3, 4]]);
        let b = mat4_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(a.mul(&b), Err("dimension mismatch"));
    }

    #[test]
    fn test_mat4_mul_jagged_left() {
        // left matrix rows have different lengths
        let a = Mat4 {
            data: vec![vec![1, 2], vec![3]], // jagged!
        };
        let b = mat4_from([[1, 2], [3, 4]]);
        assert_eq!(a.mul(&b), Err("jagged matrix"));
    }

    #[test]
    fn test_mat4_mul_jagged_right() {
        // right matrix rows have different lengths
        let a = mat4_from([[1, 2], [3, 4]]);
        let b = Mat4 {
            data: vec![vec![1, 2], vec![3]], // jagged!
        };
        assert_eq!(a.mul(&b), Err("jagged matrix"));
    }

    #[test]
    fn test_mat4_at_zero_matrix() {
        let m: Mat4<i32> = Mat4::new();
        assert_eq!(m.at(0, 0), Some(&0));
        assert_eq!(m.at(3, 3), Some(&0));
        assert_eq!(m.at(4, 0), None); // 行越界
    }
}
