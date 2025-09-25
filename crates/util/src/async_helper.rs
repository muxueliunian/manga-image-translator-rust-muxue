#[macro_export]
macro_rules! spawn_blocking {
    ($f:expr) => {{
        let (_, mut outputs) = util::async_scoped::TokioScope::scope_and_block(|s| {
            s.spawn_blocking($f);
        });
        outputs.remove(0)
    }};
}
