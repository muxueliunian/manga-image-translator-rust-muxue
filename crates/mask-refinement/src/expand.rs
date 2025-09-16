use interface_detector::textlines::MyPoint;
use interface_image::Mask;

type Quad = [MyPoint; 4];

fn vec_normalize(v: MyPoint<f64>) -> MyPoint<f64> {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len == 0.0 {
        (0.0, 0.0).into()
    } else {
        (v.x / len, v.y / len).into()
    }
}

/// $$
/// \vec{g_1}(t) = \vec{p_1} + t \cdot \vec{v_1}
/// $$
///
/// $$
/// \vec{g_2}(s) = \vec{p_2} + s \cdot \vec{v_2}
/// $$
///
/// $$
/// \vec{p_1} = (x_1, y_1), \quad \vec{v_1} = (dx_1, dy_1)
/// $$
///
/// $$
/// \vec{p_2} = (x_2, y_2), \quad \vec{v_2} = (dx_2, dy_2)
/// $$
///
/// $$
/// D = dx_1 \cdot dy_2 - dy_1 \cdot dx_2
/// $$
///
/// $$
/// t = \frac{(x_2 - x_1) \cdot dy_2 - (y_2 - y_1) \cdot dx_2}{D}
/// $$
pub fn expand_right_to_connect(quad1: &Quad, quad2: &Quad) -> Quad {
    let tl1 = quad1[0].to_f64();
    let tr1 = quad1[1].to_f64();
    let v1 = tr1 - tl1;
    let tl2 = quad2[0].to_f64();
    let bl2 = quad2[3].to_f64();
    let w = bl2 - tl2;
    let br1 = quad1[2].to_f64();
    let bl1 = quad1[3].to_f64();
    let v2 = br1 - bl1;
    let d1 = v1.x * w.y - v1.y * w.x;
    let d2 = v2.x * w.y - v2.y * w.x;
    if d1 < (10.0_f64.powi(-9)) || d2 < (10.0_f64.powi(-9)) {
        return *quad1;
    }
    let t1 = ((tl2.x - tl1.x) * w.y - (tl2.y - tl1.y) * w.x) / d1;
    let t2 = ((tl2.x - bl1.x) * w.y - (tl2.y - bl1.y) * w.x) / d2;

    let new_top = v1 * t1 + tl1;
    let new_bottom = v2 * t2 + bl1;

    [
        (tl1.x as i64, tl1.y as i64).into(),
        (new_top.x.ceil() as i64, new_top.y.ceil() as i64).into(),
        (new_bottom.x.ceil() as i64, new_bottom.y.ceil() as i64).into(),
        (bl1.x as i64, bl1.y as i64).into(),
    ]
}

pub fn expand_top_to_connect(quad1: &Quad, quad2: &Quad) -> Quad {
    let tl1 = quad1[0].to_f64();
    let tr1 = quad1[1].to_f64();
    let br1 = quad1[2].to_f64();
    let bl1 = quad1[3].to_f64();
    let v1 = tr1 - br1;
    let bl2 = quad2[3].to_f64();
    let br2 = quad2[1].to_f64();
    let w = bl2 - br2;
    let v2 = tl1 - bl1;
    let d1 = v1.x * w.y - v1.y * w.x;
    let d2 = v2.x * w.y - v2.y * w.x;
    if d1 < (10.0_f64.powi(-9)) || d2 < (10.0_f64.powi(-9)) {
        return *quad1;
    }
    let t1 = ((br2.x - br1.x) * w.y - (br2.y - br1.y) * w.x) / d1;
    let t2 = ((br2.x - bl1.x) * w.y - (br2.y - bl1.y) * w.x) / d2;

    let new_top = v1 * t1 + tl1;
    let new_bottom = v2 * t2 + bl1;

    [
        (tl1.x as i64, tl1.y as i64).into(),
        (new_top.x.ceil() as i64, new_top.y.ceil() as i64).into(),
        (new_bottom.x.ceil() as i64, new_bottom.y.ceil() as i64).into(),
        (bl1.x as i64, bl1.y as i64).into(),
    ]
}

pub fn expand_top_quad(quad: Quad, factor: f64) -> Quad {
    let (top_left, top_right, bottom_right, bottom_left) = (quad[0], quad[1], quad[2], quad[3]);

    let left_vec = top_left - bottom_left;
    let right_vec = top_right - bottom_right;

    let new_top_left = (left_vec.to_f64() * factor).to_i64() + bottom_left;
    let new_top_right = (right_vec.to_f64() * factor).to_i64() + bottom_right;

    [
        new_top_left,  // moved upward
        new_top_right, // moved upward
        bottom_right,  // unchanged
        bottom_left,   // unchanged
    ]
}

pub fn expand_right_quad(quad: Quad, factor: f64) -> Quad {
    let (top_left, top_right, bottom_right, bottom_left) = (quad[0], quad[1], quad[2], quad[3]);

    let top_vec = top_right - top_left;
    let bottom_vec = bottom_right - bottom_left;
    let new_top_right = (top_vec.to_f64() * factor).to_i64() + top_left;
    let new_bottom_right = (bottom_vec.to_f64() * factor).to_i64() + bottom_left;

    [
        top_left,         // unchanged
        new_top_right,    // expanded right
        new_bottom_right, // expanded right
        bottom_left,      // unchanged
    ]
}

fn sample_line_nonzero(mask: &Mask, start: MyPoint<f64>, end: MyPoint<f64>) -> usize {
    let (mut x0, mut y0) = (start.x as i64, start.y as i64);
    let (x1, y1) = (end.x as i64, end.y as i64);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut count = 0;
    loop {
        if mask.get(x0.max(0) as usize, y0.max(0) as usize) != 0 {
            count += 1;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
    count
}

pub fn shrink_quad_top(quad: Quad, mask: &Mask) -> Quad {
    let p1 = quad[2].to_f64(); // br
    let p2 = quad[3].to_f64(); // bl
    let mut tr = quad[1].to_f64();
    let mut tl = quad[0].to_f64();

    let v1 = tr - p1;
    let v2 = tl - p2;
    let len_v1 = (v1.x * v1.x + v1.y * v1.y).sqrt();
    let len_v2 = (v2.x * v2.x + v2.y * v2.y).sqrt();

    let half_len_v1 = len_v1 / 2.0;
    let half_len_v2 = len_v2 / 2.0;

    let dir_v1 = vec_normalize(v1);
    let dir_v2 = vec_normalize(v2);
    let step1 = dir_v1 * -1.0;
    let step2 = dir_v2 * -1.0;

    let mut moved_v1 = 0.0;
    let mut moved_v2 = 0.0;
    loop {
        if moved_v1 >= half_len_v1 || moved_v2 >= half_len_v2 {
            break;
        }
        let count = sample_line_nonzero(&mask, tr, tl);
        if count > 2 {
            break;
        }

        tr = tr + step1;
        tl = tl + step2;

        moved_v1 += 1.0;
        moved_v2 += 1.0;
    }

    if moved_v1 < half_len_v1 || moved_v2 < half_len_v2 {
        tr = tr - (step1 * 2.0);
        tl = tl - (step2 * 2.0);
    }

    [
        (tl.x as i64, tl.y as i64).into(),
        (tr.x as i64, tr.y as i64).into(),
        (p1.x as i64, p1.y as i64).into(),
        (p2.x as i64, p2.y as i64).into(),
    ]
}

pub fn shrink_quad_right(quad: Quad, mask: &Mask) -> Quad {
    let p1 = quad[0].to_f64();
    let p2 = quad[3].to_f64();
    let mut tr = quad[1].to_f64();
    let mut br = quad[2].to_f64();

    let v1 = tr - p1;
    let v2 = br - p2;
    let len_v1 = (v1.x * v1.x + v1.y * v1.y).sqrt();
    let len_v2 = (v2.x * v2.x + v2.y * v2.y).sqrt();

    let half_len_v1 = len_v1 / 2.0;
    let half_len_v2 = len_v2 / 2.0;

    let dir_v1 = vec_normalize(v1);
    let dir_v2 = vec_normalize(v2);
    let step1 = dir_v1 * -1.0;
    let step2 = dir_v2 * -1.0;

    let mut moved_v1 = 0.0;
    let mut moved_v2 = 0.0;
    loop {
        if moved_v1 >= half_len_v1 || moved_v2 >= half_len_v2 {
            break;
        }
        let count = sample_line_nonzero(&mask, tr, br);
        if count > 2 {
            break;
        }

        tr = tr + step1;
        br = br + step2;

        moved_v1 += 1.0;
        moved_v2 += 1.0;
    }

    if moved_v1 < half_len_v1 || moved_v2 < half_len_v2 {
        tr = tr - (step1 * 2.0);
        br = br - (step2 * 2.0);
    }

    [
        (p1.x as i64, p1.y as i64).into(),
        (tr.x as i64, tr.y as i64).into(),
        (br.x as i64, br.y as i64).into(),
        (p2.x as i64, p2.y as i64).into(),
    ]
}
