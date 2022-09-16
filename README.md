*This website contains information regarding the paper Modular Flows: Differential Molecular Generation.*

> **TL;DR:** We propose generative graph normalizing flow models, based on a system of coupled node ODEs, that repeatedly reconcile locally toward globally aligned densities for high quality molecular generation

# Problem of Molecular Generation

A key challenge of molecular generative models is to be able to generate valid molecules, according to various criteria for molecular validity or feasibility. It is a common practice to use external chemical software as rejection oracles to reduce or exclude invalid molecules, or do validity checks as part of autoregressive generation [1,2,3] . An important open question has been whether generative models can learn to achieve high generative validity *intrinsically*, i.e., without being aided by oracles or performing additional checks.



# Continuous Normalizing Flows

<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/nf_website.png">
</p>

Normalizing flow have seen widespread use for density modeling, generative modeling, etc which provides a general way of constructing flexible probability distributions. It is defined by a parameterized invertible deterministic transformation from a base distribution $$\mathcal{Z}$$ (e.g., Gaussian distribution) to real-world observational space $$X$$ (e.g. images and speech). When the dynamics of transformation is governed by an ODE, the method is known as Continous Normalizing Flows (CNFs). The process starts by sampling from a base distribution $$\mathbf{z}_0 \sim p_0(\mathbf{z}_0)$$, then solving the IVP $$\mathbf{z}(t_0) = \mathbf{z}_0$$, $$\dot{\mathbf{z}}(t) = \frac{\partial \mathbf{z}(t)}{\partial t} = f(\mathbf{z}(t),t;\theta)$$, where ODE is defined by the parametric function $$f(\mathbf{z}(t),t;\theta)$$ to obtain $$\mathbf{z}(t_1)$$ which constitutes our observable data. Then, using the *instantaneous change of variables* formula change in log-density under this model is given as:


<p align="center">
    $$\frac{\partial \log p_t(\mathbf{z}(t))}{\partial t} = -\texttt{tr} \left( \frac{\partial f}{\partial \mathbf{z}(t)} \right)$$
</p> 

Given a datapoint $$\mathbf{x}$$, we can compute both the point $$\mathbf{z}_{0}$$ which generates $$\mathbf{x}$$, as well as $$\log p_1(\mathbf{x})$$ by solving the initial value problem which integrates the combined dynamics of $$\mathbf{z}(t)$$ and the log-density of the sample resulting in the computation of $$\log p_{1}(\mathbf{x})$$.


# Modular Flows

## Representation

We represent molecule as a graph $$G = (V,E)$$, where each vertex takes value from an alphabet on atoms:  $$v \in \mathcal{A} = \{ \texttt{C},\texttt{H},\texttt{N},\texttt{O},\texttt{P},\texttt{S},\ldots \}$$; while the edges $$e \in \mathcal{B} = \{1,2,3\}$$ abstract the type of bond (i.e., single, double, or triple). We assume the following decomposition of the graph likelihood, over vertices conditioned on the edges and given the latent representations, 

<p align="center">
    $$p(G) := p(V | E,\{ z\}) = \prod_{i=1}^M \texttt{Cat}(v_i | \sigma(\mathbf{z}_i))$$
</p> 

We obtained an alternative representation by decomposing a molecule into a tree like structure, by contracting certain vertices into a single node such that the molecular graph $$G$$ becomes acyclic. We followed a similar decompositon as JT-VAE[7], but restrict these clusters to ring-substructures, in addition to the atom alphabet. Thus, we obtain an extended alphabet vocabulary as $$\mathcal{A}_{\mathrm{tree}} = \{ \texttt{C},\texttt{H},\texttt{N}, \ldots,  \texttt{C}_{1},\texttt{C}_{2},\ldots \}$$, where each cluster label $$\texttt{C}_{r}$$ corresponds to the some ring-substructure in the label vocabulary $$\chi$$.

<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/junction_mod.png">
</p>




# References

[1]
[2]
[3]
[4]
[5]
[6]





