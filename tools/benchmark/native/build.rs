include!("../../../runtime/rust_link_helper.rs");

fn main() {
    emit_link_directives(&["xinfer_benchmark_shim"]);
}
