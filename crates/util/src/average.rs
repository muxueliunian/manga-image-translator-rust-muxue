use std::ops::AddAssign;

#[derive(Default, Copy, Clone)]
pub struct AvgMeter<T: Default + Copy + Clone> {
    sum: T,
    count: usize,
}

impl<T: Default + Copy + Clone + AddAssign + Into<f64>> AvgMeter<T> {
    pub fn new() -> Self {
        Self {
            sum: Default::default(),
            count: 0,
        }
    }

    pub fn update(&mut self, value: T) {
        self.sum += value;
        self.count += 1;
    }

    pub fn average(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum.into() / self.count as f64
        }
    }

    pub fn reset(&mut self) {
        self.sum = Default::default();
        self.count = 0;
    }
}
