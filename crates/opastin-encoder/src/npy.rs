//! Minimal writer for NumPy `.npy` files (format version 1.0, little-endian, C order),
//! so Python probe training can `np.load` what the Rust encoder extracts.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub trait NpyElement: Copy {
    const DESCR: &'static str;
    fn write_le(self, w: &mut impl Write) -> io::Result<()>;
}

impl NpyElement for f32 {
    const DESCR: &'static str = "<f4";
    fn write_le(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}

impl NpyElement for f64 {
    const DESCR: &'static str = "<f8";
    fn write_le(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}

impl NpyElement for i64 {
    const DESCR: &'static str = "<i8";
    fn write_le(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}

/// Write `data` with the given shape. `data.len()` must equal the product of `shape`.
pub fn write<T: NpyElement>(path: impl AsRef<Path>, shape: &[usize], data: &[T]) -> io::Result<()> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "shape {shape:?} needs {expected} elements, got {}",
                data.len()
            ),
        ));
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&header(T::DESCR, shape))?;
    for &x in data {
        x.write_le(&mut w)?;
    }
    w.flush()
}

fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // magic (6) + version (2) + header length (2) + dict, padded to a multiple of 64.
    let unpadded = 10 + dict.len() + 1;
    dict.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    dict.push('\n');

    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_aligned_and_well_formed() {
        let h = header("<f4", &[2, 3, 4]);
        assert_eq!(h.len() % 64, 0);
        assert!(h.starts_with(b"\x93NUMPY\x01\x00"));
        let text = std::str::from_utf8(&h[10..]).unwrap();
        assert!(text.contains("'shape': (2, 3, 4)"));
        assert!(text.ends_with('\n'));
        assert!(std::str::from_utf8(&header("<i8", &[5])[10..])
            .unwrap()
            .contains("(5,)"));
    }

    #[test]
    fn rejects_mismatched_shape() {
        let dir = std::env::temp_dir().join("opastin-npy-test.npy");
        assert!(write(&dir, &[2, 2], &[1.0f32, 2.0, 3.0]).is_err());
    }
}
