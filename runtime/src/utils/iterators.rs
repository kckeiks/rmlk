#[derive(Clone)]
pub struct DataIterator<'a, T> {
    shape: &'a [usize],
    stride: &'a [usize],
    data: &'a [T],
    current: usize,
}

impl<'a, T> DataIterator<'a, T> {
    pub fn new(shape: &'a [usize], stride: &'a [usize], data: &'a [T]) -> Self {
        debug_assert_eq!(shape.len(), stride.len());

        Self {
            shape,
            stride,
            data,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for DataIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.data.len() {
            return None;
        }

        let old_current = self.current;
        self.current += 1;

        match self.shape.len() {
            0 => self.data.get(old_current),
            _ => {
                let mut i = 0;
                let mut tmp_i = old_current;
                for (d_i, dim) in self.shape.iter().enumerate().rev() {
                    let norm_i = tmp_i % dim;
                    i += norm_i * self.stride[d_i];
                    tmp_i /= dim;
                }

                self.data.get(i)
            }
        }
    }
}
