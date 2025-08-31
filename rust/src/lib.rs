use pyo3::prelude::*;

#[pyfunction]
pub fn abc() {
    println!("hello from abc");
}

#[pymodule]
fn rblaidd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(abc, m)?)?;
    Ok(())
}