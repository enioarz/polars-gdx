//! Checked byte counts shared by adapter admission paths.

pub(super) fn allocation_bytes(parts: &[(usize, usize)]) -> Option<usize> {
    parts.iter().try_fold(0usize, |total, &(count, width)| {
        total.checked_add(count.checked_mul(width)?)
    })
}

#[cfg(test)]
mod tests {
    use super::allocation_bytes;

    #[test]
    fn layouts_reject_product_and_sum_overflow() {
        assert_eq!(allocation_bytes(&[(usize::MAX, 2)]), None);
        assert_eq!(allocation_bytes(&[(usize::MAX, 1), (1, 1)]), None);
        assert_eq!(allocation_bytes(&[(usize::MAX, 0), (4, 8)]), Some(32));
        assert_eq!(allocation_bytes(&[]), Some(0));
    }
}
