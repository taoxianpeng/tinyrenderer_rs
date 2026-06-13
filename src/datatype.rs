use std::fmt;

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
}
