use ndarray::{Array, Dimension};

pub fn to_raw<T: Clone, D: Dimension, Out>(
    arr: &Array<T, D>,
    convert: impl Fn(&[T]) -> Out,
) -> Out {
    if let Some(v) = arr.as_slice() {
        convert(v)
    } else {
        convert(
            arr.as_standard_layout()
                .as_slice()
                .expect("Failed to convert to standard layout"),
        )
    }
}
