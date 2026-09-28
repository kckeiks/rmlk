//! Float comparison helpers for rmlk vs ORT outputs.

/// Max `|a - e|` over paired elements, or `None` if lengths differ.
pub fn max_abs_diff(actual: &[f32], expected: &[f32]) -> Option<f32> {
    if actual.len() != expected.len() {
        return None;
    }
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// Element-wise closeness: `|a - e| <= atol + rtol * |e|`.
///
/// On failure, reports the index with the largest `abs_err / tol` ratio among
/// elements that exceeded tolerance (not merely the largest absolute error).
pub fn assert_close_named(
    output: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{output}: length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut worst_i = 0usize;
    let mut worst_abs = 0.0f32;
    let mut worst_tol = 0.0f32;
    let mut worst_a = 0.0f32;
    let mut worst_e = 0.0f32;
    let mut worst_ratio = 0.0f32;
    let mut failed = false;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        let abs = (a - e).abs();
        if abs > tol {
            failed = true;
            let ratio = if tol > 0.0 { abs / tol } else { f32::INFINITY };
            if ratio >= worst_ratio {
                worst_ratio = ratio;
                worst_abs = abs;
                worst_i = i;
                worst_tol = tol;
                worst_a = a;
                worst_e = e;
            }
        }
    }
    if failed {
        return Err(format!(
            "{output}: tolerance exceeded at index {worst_i}: \
             actual={worst_a} expected={worst_e} abs_err={worst_abs} tol={worst_tol} \
             (atol={atol} rtol={rtol})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_within_tolerance() {
        assert_close_named("y", &[1.0, 2.0], &[1.0, 2.00001], 1e-4, 1e-4).unwrap();
    }

    #[test]
    fn reports_worst_ratio_failure_not_largest_abs() {
        // Index 0: large abs but within tol (large |e|).
        // Index 1: small abs but exceeds tiny tol around ~0.
        let err = assert_close_named(
            "logits",
            &[10.0, 0.002],
            &[10.0, 0.0],
            1e-3,
            1e-2,
        )
        .unwrap_err();
        assert!(err.contains("index 1"), "{err}");
        assert!(err.contains("abs_err=0.002"), "{err}");
    }

    #[test]
    fn reports_worst_index_and_tol() {
        let err = assert_close_named("logits", &[0.0, 10.0], &[0.0, 0.0], 1e-5, 0.0).unwrap_err();
        assert!(err.contains("logits:"), "{err}");
        assert!(err.contains("index 1"), "{err}");
        assert!(err.contains("tol="), "{err}");
    }

    #[test]
    fn length_mismatch() {
        let err = assert_close_named("y", &[1.0], &[1.0, 2.0], 1e-5, 1e-5).unwrap_err();
        assert!(err.contains("length mismatch"), "{err}");
    }
}
