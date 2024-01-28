use std::arch::x86_64::*;

trait Vec256 {
    fn to_underlying(&self) -> __m256i;
    fn from_underlying(repr: __m256i) -> Self where Self: Sized;
}

// any type that is integer-based will have constructors and bitwise operations
trait Vec256base: Vec256 + Sized {
    fn new() -> Self where Self: Sized {
        unsafe { std::mem::zeroed() }
    }
    fn from_slice(slice: &[u8]) -> Self where Self: Sized {
        assert_eq!(slice.len(), 32);
        let repr = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
        Self::from_underlying(repr)
    }
    fn xor(&self, other: &Self) -> Self {
        let repr = unsafe { _mm256_xor_si256(self.to_underlying(), other.to_underlying()) };
        Self::from_underlying(repr)
    }
    fn and(&self, other: &Self) -> Self {
        let repr = unsafe { _mm256_and_si256(self.to_underlying(), other.to_underlying()) };
        Self::from_underlying(repr)
    }
}

#[derive(Debug)]
struct Vec32i8 {
    data: __m256i,
}

impl Vec256 for Vec32i8 {
    fn to_underlying(&self) -> __m256i {
        self.data
    }
    fn from_underlying(repr: __m256i) -> Self where Self: Sized {
        Vec32i8 { data: repr }
    }
}

impl PartialEq for Vec32i8 {
    fn eq(&self, other: &Self) -> bool {
        let repr = unsafe { _mm256_cmpeq_epi8(self.to_underlying(), other.to_underlying()) };
        unsafe { _mm256_movemask_epi8(repr) == 0xFFFF_FFFFu32 as i32 }
    }
}

impl Vec256base for Vec32i8 {}

impl Vec32i8 {
    fn add(&self, other: &Self) -> Self {
        let repr = unsafe { _mm256_add_epi8(self.to_underlying(), other.to_underlying()) };
        Self::from_underlying(repr)
    }
    fn sub(&self, other: &Self) -> Self {
        let repr = unsafe { _mm256_sub_epi8(self.to_underlying(), other.to_underlying()) };
        Self::from_underlying(repr)
    }
}

struct Vec32i8fallback {
    data: [u8; 32],
}

impl Vec32i8fallback {
    fn new() -> Self {
        Vec32i8fallback { data: [0; 32] }
    }
    fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(slice.len(), 32);
        let mut data = [0; 32];
        data.copy_from_slice(slice);
        Vec32i8fallback { data }
    }
    fn xor(&self, other: &Self) -> Self {
        let mut data = [0; 32];
        for i in 0..32 {
            data[i] = self.data[i] ^ other.data[i];
        }
        Vec32i8fallback { data }
    }
    fn and(&self, other: &Self) -> Self {
        let mut data = [0; 32];
        for i in 0..32 {
            data[i] = self.data[i] & other.data[i];
        }
        Vec32i8fallback { data }
    }
    fn add(&self, other: &Self) -> Self {
        let mut data = [0; 32];
        for i in 0..32 {
            data[i] = self.data[i].wrapping_add(other.data[i]);
        }
        Vec32i8fallback { data }
    }
    fn sub(&self, other: &Self) -> Self {
        let mut data = [0; 32];
        for i in 0..32 {
            data[i] = self.data[i].wrapping_sub(other.data[i]);
        }
        Vec32i8fallback { data }
    }

    fn eq(&self, other: &Self) -> bool {
        for i in 0..32 {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true
    }

    fn eq_to_vec32i8(&self, other: &Vec32i8) -> bool {
        let mut data = [0; 32];
        unsafe { _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, other.to_underlying()) };
        for i in 0..32 {
            if self.data[i] != data[i] {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec256i8_works() {
        {
            let result_new = Vec32i8::new();
            assert_eq!(result_new, Vec32i8::new());
        }
        {
            let result_basics = Vec32i8::from_slice(&[
                255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255
            ]);
            assert_ne!(result_basics, Vec32i8::new());
            let xor_with_self = result_basics.xor(&result_basics);
            assert_eq!(xor_with_self, Vec32i8::new());
            let and_with_self = result_basics.and(&result_basics);
            assert_eq!(and_with_self, result_basics);
        }
        {
            let result_add = Vec32i8::from_slice(&[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            ]);
            let result_add2 = Vec32i8::from_slice(&[
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
            ]);
            let result_add3 = result_add.add(&result_add2);
            assert_eq!(result_add3, Vec32i8::from_slice(&[
                2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16,
                18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 32, 32
            ]));
            let result_sub = result_add.sub(&result_add2);
            assert_eq!(result_sub, Vec32i8::from_slice(&[
                0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16,
                16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 32
            ]));
        }
    }

    fn random_slice() -> [u8; 32] {
        let mut x = [0; 32];
        for i in 0..32 {
            x[i] = rand::random::<u8>();
        }
        x
    }

    #[test]
    fn vec256i8_compare_to_fallback() {
        for _ in 0..1000 {
            let x = random_slice();
            let y = random_slice();
            let result = Vec32i8::from_slice(&x).add(&Vec32i8::from_slice(&y));
            let result_fallback = Vec32i8fallback::from_slice(&x).add(&Vec32i8fallback::from_slice(&y));
            assert!(result_fallback.eq_to_vec32i8(&result));
        }

        for _ in 0..1000 {
            let x = random_slice();
            let y = random_slice();
            let result = Vec32i8::from_slice(&x).sub(&Vec32i8::from_slice(&y));
            let result_fallback = Vec32i8fallback::from_slice(&x).sub(&Vec32i8fallback::from_slice(&y));
            assert!(result_fallback.eq_to_vec32i8(&result));
        }

        for _ in 0..1000 {
            let x = random_slice();
            let y = random_slice();
            let result = Vec32i8::from_slice(&x).xor(&Vec32i8::from_slice(&y));
            let result_fallback = Vec32i8fallback::from_slice(&x).xor(&Vec32i8fallback::from_slice(&y));
            assert!(result_fallback.eq_to_vec32i8(&result));
        }
    }
}