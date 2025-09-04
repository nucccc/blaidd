use rand::rng;
use rand::seq::SliceRandom;

use pyo3::prelude::*;


#[derive(Clone)]
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

    fn get_n_centroids(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone)]
pub struct Node {
    connecteds: Vec<usize>,
    color: f64, // maybe not option at a point, like let's say we'll have -1
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

        Self { connecteds: connecteds, color: -1.0 }
    }
}


fn create_scale(n_centroids: usize) -> Vec<f64> {
    // TODO: test this
    let mut res = Vec::new();
    res.reserve(n_centroids);

    let step: f64 = 1.0f64 / ((n_centroids-1) as f64);

    for i in 0..(n_centroids-1) {
        res.push((i as f64) * step);
    }
    res.push(1.0);

    res
}


struct Bee {
    graph: Graph,
    fitness: f64,
    no_improv: u32,
    scale: Vec<f64>,
}

impl Bee {
    fn new(graph: Graph) -> Self {
        let scale = create_scale(graph.get_n_centroids());

        let mut bee = Bee{
            graph: graph,
            fitness: 0.0,
            no_improv: 0,
            scale: scale,
        };

        bee.reset();

        bee
    }

    fn reset(&mut self, ) {
        self.scale.shuffle(&mut rng());
        /*for i in 0.. {
            node.color = self.scale[i]
        }*/
        for (node, color) in self.graph.nodes.iter_mut().zip(self.scale.iter()) {
            node.color = *color
        }
    }
}


#[pyfunction]
pub fn abc(
    n_centroids: usize,
    edges: Vec<(usize, usize)>,
    n_bees: usize,
) {
    println!("hello from abc");
    println!("{:?}", edges);

    let mut graph = Graph::new(n_centroids, &edges);

    let scale = create_scale(n_centroids);

    let mut bees: Vec<Bee> = Vec::new();

    for _ in 0..n_bees {
        bees.push(Bee::new(graph.clone()));
    } 


}

#[pymodule]
fn rblaidd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(abc, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        let scale0 = create_scale(2);
        assert_eq!(scale0, vec![0.0, 1.0]);

        let scale1 = create_scale(3);
        assert_eq!(scale1, vec![0.0, 0.5, 1.0]);

        let scale1 = create_scale(4);
        assert_eq!(scale1, vec![0.0, 0.3333333333333333, 0.6666666666666666, 1.0]);

        let scale2 = create_scale(5);
        assert_eq!(scale2, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
