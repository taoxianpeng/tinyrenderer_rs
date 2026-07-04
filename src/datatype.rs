pub use glam::Vec3;
pub use glam::IVec2 as Point2D;



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Point2D::new(10, 30);
        let b = Point2D::new(20, 40);
        assert_eq!(a + b, Point2D::new(30, 70));
    }

    #[test]
    fn test_add_assign() {
        let mut a = Point2D::new(10, 30);
        a += Point2D::new(20, 40);
        assert_eq!(a, Point2D::new(30, 70));
    }

    #[test]
    fn test_sub() {
        let a = Point2D::new(10, 30);
        let b = Point2D::new(20, 40);
        assert_eq!(a - b, Point2D::new(-10, -10));
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Point2D::new(10, 30);
        a -= Point2D::new(20, 40);
        assert_eq!(a, Point2D::new(-10, -10));
    }

    #[test]
    fn test_mul_scalar() {
        let a = Point2D::new(10, 30);
        assert_eq!(a * 2, Point2D::new(20, 60));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut a = Point2D::new(10, 30);
        a *= 2;
        assert_eq!(a, Point2D::new(20, 60));
    }

    #[test]
    fn test_div_scalar() {
        let a = Point2D::new(10, 30);
        assert_eq!(a / 2, Point2D::new(5, 15));
    }

    #[test]
    fn test_div_assign_scalar() {
        let mut a = Point2D::new(10, 30);
        a /= 2;
        assert_eq!(a, Point2D::new(5, 15));
    }

    #[test]
    fn test_distance() {
        let a = Point2D::new(0, 0);
        let b = Point2D::new(3, 4);
        assert!((a.as_vec2().distance(b.as_vec2()) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_field_access() {
        let p = Point2D::new(10, 20);
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);
    }
}
