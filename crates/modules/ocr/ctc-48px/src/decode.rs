use ndarray::{Array, Array4, ArrayD, Axis};
use ndarray_stats::QuantileExt as _;
use ort::{inputs, session::Session, value::Tensor};

pub fn decode(
    model: &mut Session,
    img: Array4<f32>,
    blank: i64,
) -> Vec<Vec<(i64, f32, f32, f32, f32, f32, f32, f32)>> {
    let out = model
        .run(inputs! {"images"=> Tensor::from_array(img).unwrap()})
        .unwrap();
    let pred_char_logits = out.get("pred_char_logits").unwrap();
    let mut pred_chars = (0..pred_char_logits.shape()[0])
        .map(|_| vec![])
        .collect::<Vec<_>>();
    let logprobs = log_softmax(
        pred_char_logits.try_extract_array().unwrap().to_owned(),
        Axis(2),
    );
    let (_, preds_index) = max_along_axis(&logprobs, Axis(2));
    let pred_color_values = out
        .get("pred_color_values")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap()
        .mapv(|v| v.clamp(0.0, 1.0));
    for b in 0..pred_char_logits.shape()[0] as usize {
        let mut last_ch = blank;
        for t in 0..pred_char_logits.shape()[1] as usize {
            let pred_ch: i64 = preds_index[[b, t]];
            if pred_ch != last_ch && pred_ch != blank {
                let lp: f32 = logprobs[[b, t, pred_ch as usize]];
                pred_chars[b as usize].push((
                    pred_ch,
                    lp,
                    pred_color_values[[b, t, 0]],
                    pred_color_values[[b, t, 1]],
                    pred_color_values[[b, t, 2]],
                    pred_color_values[[b, t, 3]],
                    pred_color_values[[b, t, 4]],
                    pred_color_values[[b, t, 5]],
                ));
            }
            last_ch = pred_ch;
        }
    }
    pred_chars
}

fn max_along_axis(arr: &ArrayD<f32>, axis: Axis) -> (ArrayD<f32>, Array<i64, ndarray::IxDyn>) {
    let max_vals = arr.map_axis(axis, |view| {
        *view
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    });

    let argmax_indices = arr.map_axis(axis, |view| view.argmax().unwrap() as i64);

    (max_vals.into_dyn(), argmax_indices.into_dyn())
}

fn log_softmax(mut arr: ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
    let max = arr.map_axis(axis, |v| v.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    arr = arr - &max.insert_axis(axis);
    let exp = arr.mapv(|x| x.exp());
    let sum_exp = exp.sum_axis(axis).insert_axis(axis);
    &arr - &sum_exp.mapv(|x| x.ln())
}
