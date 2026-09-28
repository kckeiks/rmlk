//! ImageNet-style preprocessing for vision e2e cases.

use image::imageops::{crop_imm, resize, FilterType};
use image::RgbImage;
use std::path::Path;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Torchvision ResNet eval preprocess: RGB → shorter side 256 → center crop 224
/// → NCHW float32 normalized with ImageNet mean/std.
pub fn imagenet_resnet_nchw(path: &Path) -> Result<(Vec<f32>, Vec<usize>), String> {
    let img = image::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(format!("{}: empty image", path.display()));
    }
    let scale = 256.0 / (w.min(h) as f32);
    let nw = ((w as f32) * scale).round().max(1.0) as u32;
    let nh = ((h as f32) * scale).round().max(1.0) as u32;
    let resized: RgbImage = resize(&img, nw, nh, FilterType::Triangle);
    if nw < 224 || nh < 224 {
        return Err(format!(
            "{}: resized {nw}x{nh} smaller than 224 crop",
            path.display()
        ));
    }
    let left = (nw - 224) / 2;
    let top = (nh - 224) / 2;
    let cropped = crop_imm(&resized, left, top, 224, 224).to_image();

    let mut data = vec![0.0f32; 1 * 3 * 224 * 224];
    for y in 0..224 {
        for x in 0..224 {
            let p = cropped.get_pixel(x, y);
            for c in 0..3 {
                let v = f32::from(p[c]) / 255.0;
                let norm = (v - MEAN[c]) / STD[c];
                data[c * 224 * 224 + (y as usize) * 224 + (x as usize)] = norm;
            }
        }
    }
    Ok((data, vec![1, 3, 224, 224]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_dog() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/e2e/resnet34/dog.jpeg")
    }

    #[test]
    fn dog_jpeg_preprocesses_to_nchw() {
        let (data, shape) = imagenet_resnet_nchw(&fixture_dog()).unwrap();
        assert_eq!(shape, vec![1, 3, 224, 224]);
        assert_eq!(data.len(), 1 * 3 * 224 * 224);
        // Rough sanity: normalized values are not all zero and not huge.
        let max = data.iter().cloned().fold(0.0f32, f32::max);
        let min = data.iter().cloned().fold(0.0f32, f32::min);
        assert!(max < 5.0 && min > -5.0, "min={min} max={max}");
    }
}
