#![no_main]
#[path = "common.rs"]
mod common;
#[path = "qp_body.rs"]
mod qp_body;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| qp_body::run(data, false));
