use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};


pub struct Graph {
    nodes: Vec<Node>
}

impl Graph {
    fn new(n_centroids: usize, edges: &Vec<(usize, usize)>) -> Self {
        let mut nodes: Vec<Node> = Vec::new();

        for node_index in 0..n_centroids {
            nodes.push(Node::new(node_index, edges));
        }
        
        Self { nodes: nodes }
    }
}

pub struct Node {
    connecteds: Vec<usize>,
    color: Option<f64>, // maybe not option at a point, like let's say we'll have -1
}

impl Node {
    fn new(node_index: usize, edges: &Vec<(usize, usize)>) -> Self {
        let mut connecteds = Vec::new();
        
        for &(n1, n2) in edges {
            if n1 == node_index {
                connecteds.push(n2);
            } else if n2 == node_index {
                connecteds.push(n1);
            }
        }

        Self { connecteds: connecteds, color: Some(-1.0) }
    }
}


#[pyfunction]
pub fn abc(
    n_centroids: usize,
    edges: Vec<(usize, usize)>
) {
    println!("hello from abc");
    println!("{:?}", edges);

    let mut graph = Graph::new(n_centroids, &edges);

    /*for elem in l {
        let boh: PyTuple = elem.extract().unwrap();
        println!("{boh}");
    }*/
}

#[pymodule]
fn rblaidd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(abc, m)?)?;
    Ok(())
}