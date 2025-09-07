use rand::rng;
use rand::seq::SliceRandom;

use pyo3::prelude::*;


type Solution = Vec<f64>;


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

        Self { connecteds: connecteds }
    }
}


fn create_scale(n_centroids: usize) -> Solution {
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
    n_stuck: u32,
    max_stuck: u32,
    solution: Solution,
    max_diff: f64,
}

/*  max_diff suppesed to be between 0.0 and 1.0 */
fn calc_fitness(graph: &Graph, solution: &Solution, max_diff: f64) -> f64 {
    let mut fitness = 0.0;

    for (i, node) in graph.nodes.iter().enumerate() {
        if ! node.connecteds.is_empty() {
            let mut node_score = 0.0;
            for j in &node.connecteds {
                let diff = (solution[i] - solution[*j]).abs();
                let diff_norm = f64::min(diff, max_diff);
                node_score += diff_norm.powi(2)
            }
            fitness += node_score / (node.connecteds.len() as f64);
        }        
    }
    fitness = fitness / (graph.nodes.len() as f64);

    fitness
}

fn gen_perm(max_val: usize) -> (usize, usize) {
    let i = rand::random_range(..max_val);
    let mut j = rand::random_range(..max_val);

    while i == j {
        j = rand::random_range(..max_val);
    }

    (i, j)
}

fn gen_perms(n_perms: usize, max_val: usize) -> Vec<(usize, usize)> {
    (0..n_perms).map(|_| gen_perm(max_val)).collect()
}

impl Bee {
    fn new(graph: Graph, max_diff: f64, max_stuck: u32) -> Self {
        // instantiating a scale, which will be the solution
        let scale = create_scale(graph.get_n_centroids());

        let mut bee = Bee{
            graph: graph,
            fitness: 0.0,
            n_stuck: 0,
            max_stuck: max_stuck,
            solution: scale,
            max_diff: max_diff,
        };

        // resetting in order for the solution to be randomized
        bee.reset();

        bee
    }

    fn reset(&mut self) {
        self.solution.shuffle(&mut rng());

        self.n_stuck = 0;
        self.fitness = calc_fitness(&self.graph, &self.solution, self.max_diff);
    }

    fn iterate(&mut self, n_perms: usize) {
        // TODO: this could become an internal array at a point for performance reasons
        let perms = gen_perms(n_perms, self.solution.len());
    
        self.apply_perms(&perms);

        let new_fitness = calc_fitness(&self.graph, &self.solution, self.max_diff);
        if new_fitness > self.fitness {
            self.fitness = new_fitness;
            self.n_stuck = 0;
        } else {
            self.n_stuck += 1;
            if self.n_stuck >= self.max_stuck {
                self.reset();
            } else {
                self.reset_perms(&perms);
            }
        }
    }

    fn apply_perms(&mut self, perms: &Vec<(usize, usize)>) {
        for perm in perms {
            let c = self.solution[perm.0];
            self.solution[perm.0] = self.solution[perm.1];
            self.solution[perm.1] = c;
        }
    }

    fn reset_perms(&mut self, perms: &Vec<(usize, usize)>) {
        for perm in perms.iter().rev() {
            let c = self.solution[perm.0];
            self.solution[perm.0] = self.solution[perm.1];
            self.solution[perm.1] = c;
        }
    }
}


#[pyfunction]
pub fn abc(
    n_centroids: usize,
    edges: Vec<(usize, usize)>,
    n_bees: usize,
    n_iter: usize, // TODO: this could become a time thing at a point
) -> Solution {
    // TODO: this shall become a parameter with a default
    let n_perms: usize = 1;

    let mut best_fitness: f64;
    let mut best_solution: Solution;

    // TODO: this shall be passed with references at a point
    let graph = Graph::new(n_centroids, &edges);

    // instantiating bees
    let mut bees: Vec<Bee> = Vec::new();
    for _ in 0..n_bees {
        bees.push(Bee::new(
            graph.clone(),
            0.25,
            50
        ));
    }

    /*
    gotta select the best solution as the first bee, anyways the
    subsequent ones will be evaluated once
    */

    best_fitness = bees.first().unwrap().fitness;
    best_solution = bees.first().unwrap().solution.clone();

    for _ in 0..n_iter {
        // performing round of iteration
        for bee in bees.iter_mut() {
            bee.iterate(n_perms);
        }
        // selecting best solution
        for bee in bees.iter() {
            // this is a maximization problem
            if best_fitness < bee.fitness {
                best_solution = bee.solution.clone();
                best_fitness = bee.fitness
            }
        }
    }

    best_solution
}

#[pymodule]
fn _rblaidd(_py: Python, m: &PyModule) -> PyResult<()> {
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
