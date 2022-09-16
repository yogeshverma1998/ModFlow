*This website contains information regarding the paper Modular Flows: Differential Molecular Generation.*

> **TL;DR:** We propose generative graph normalizing flow models, based on a system of coupled node ODEs, that repeatedly reconcile locally toward globally aligned densities for high quality molecular generation

# Problem of Molecular Generation

A key challenge of molecular generative models is to be able to generate valid molecules, according to various criteria for molecular validity or feasibility. It is a common practice to use external chemical software as rejection oracles to reduce or exclude invalid molecules, or do validity checks as part of autoregressive generation [1,2,3] . An important open question has been whether generative models can learn to achieve high generative validity *intrinsically*, i.e., without being aided by oracles or performing additional checks.



# Continuous Normalizing Flows

<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/nf_website.png" />
</p>

Normalizing flow have seen widespread use for density modeling, generative modeling, etc which provides a general way of constructing flexible probability distributions. It is defined by a parameterized invertible deterministic transformation from a base distribution $$\mathcal{Z}$$ (e.g., Gaussian distribution) to real-world observational space $$X$$ (e.g. images and speech). When the dynamics of transformation is governed by an ODE, the method is known as Continous Normalizing Flows (CNFs). The process starts by sampling from a base distribution $$\mathbf{z}_0 \sim p_0(\mathbf{z}_0)$$, then solving the IVP $$\mathbf{z}(t_0) = \mathbf{z}_0$$, $$\dot{\mathbf{z}}(t) = \frac{\partial \mathbf{z}(t)}{\partial t} = f(\mathbf{z}(t),t;\theta)$$, where ODE is defined by the parametric function $$f(\mathbf{z}(t),t;\theta)$$ to obtain $$\mathbf{z}(t_1)$$ which constitutes our observable data. Then, using the *instantaneous change of variables* formula change in log-density under this model is given as:

<p align="center">
    $$\frac{\partial \log p_t(\mathbf{z}(t))}{\partial t} = -\texttt{tr} \left( \frac{\partial f}{\partial \mathbf{z}(t)} \right)$$
 </p> 

Given a datapoint $$\mathbf{x}$$, we can compute both the point $$\mathbf{z}_{0}$$ which generates $$\mathbf{x}$$, as well as $$\log p_1(\mathbf{x})$$ by solving the initial value problem which integrates the combined dynamics of $$\mathbf{z}(t)$$ and the log-density of the sample resulting in the computation of $$\log p_1(\mathbf{x})$$.




<!--The $$f: \mathcal{Z} \mapsto X$$ is an invertible transformation, then we can compute the density function of real-world data $$\mathbf{x}$$, i.e., $$p_X(\mathbf{x})$$, via the change-of-variables formula:

<p align="center">
$$p_X(\mathbf{x}) = p_{\mathcal{Z}}\big(f_{\theta}^{-1}(\mathbf{x}) \big) \left| \det \frac{\partial f_{\theta}^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|$$
 </p>   
Given a datapoint $$\mathbf{x}$$, the exact density $$p_X(\mathbf{x})$$ can be computed via inverting the flow by function $$f$$, $$\mathbf{z} = f^{-1}(\mathbf{x})$$. Moreover, $$\mathbf{x}$$ can be sampled from $$p_X(\mathbf{x})$$ by first sampling $$\mathbf{z} \sim p_\mathcal{Z}(\mathbf{z})$$ and then performing the feedforward transformation $$\mathbf{x} = f_{\theta}(\mathbf{z})$$. 



There exists a continous analog of above equation which replaces the  warping function with an integral of continous-time dynamics. The process starts by sampling from a base distribution $$\mathbf{z}_0 \sim p_0(\mathbf{z}_0)$$. Then, we solve the initial value problem $$\mathbf{z}(t_0) = \mathbf{z}_0$$, $$\dot{\mathbf{z}}(t) = \frac{\partial \mathbf{z}(t)}{\partial t} = f(\mathbf{z}(t),t;\theta)$$, where ODE is defined by the parametric function $$f(\mathbf{z}(t),t;\theta)$$ to obtain $$\mathbf{z}(t_1)$$ which constitutes our observable data. These models are called Continous Normalizing Flows (CNF). Then, using the *instantaneous change of variables* formula change in log-density under this model is given as:

<p align="center">
    $$\frac{\partial \log p_t(\mathbf{z}(t))}{\partial t} = -\texttt{tr} \left( \frac{\partial f}{\partial \mathbf{z}(t)} \right)$$
 </p> 

Given a datapoint $$\mathbf{x}$$, we can compute both the point $$\mathbf{z}_{0}$$ which generates $$\mathbf{x}$$, as well as $$\log p_1(\mathbf{x})$$ by solving the initial value problem which integrates the combined dynamics of $$\mathbf{z}(t)$$ and the log-density of the sample resulting in the computation of $$\log p_1(\mathbf{x})$$.-->




# Modular Flows

## Representation

We represent molecule as a graph $$G = (V,E)$$, where each vertex takes value from an alphabet on atoms:  $$v \in \mathcal{A} = \{ \texttt{C},\texttt{H},\texttt{N},\texttt{O},\texttt{P},\texttt{S},\ldots \}$$; while the edges $$e \in \mathcal{B} = \{1,2,3\}$$ abstract the type of bond (i.e., single, double, or triple). We assume the following decomposition of the graph likelihood, over vertices conditioned on the edges and given the latent representations, 

<p align="center">
    $$p(G) := p(V | E,\{ z\}) = \prod_{i=1}^M \texttt{Cat}(v_i | \sigma(\mathbf{z}_i))$$
  </p> 

We obtained an alternative representation by decomposing a molecule into a tree like structure, by contracting certain vertices into a single node such that the molecular graph $$G$$ becomes acyclic. We followed a similar decompositon as JT-VAE[7], but restrict these clusters to ring-substructures, in addition to the atom alphabet. Thus, we obtain an extended alphabet vocabulary as $$\mathcal{A}_{\mathrm{tree}} = \{ \texttt{C},\texttt{H},\texttt{N}, \ldots,  \texttt{C}_{1},\texttt{C}_{2},\ldots \}$$, where each cluster label $$\texttt{C}_{r}$$ corresponds to the some ring-substructure in the label vocabulary $$\chi$$.

<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/junction_mod.png" />
</p>


## Differential Modular Flows

Based on the general recipie of normalizing flows, we propose to model the node scores $$\mathbf{z}_{i}$$ as a Continuous-time Normalizing Flow (CNF)[4] over time $$t \in \mathrm{R}_+$$. We assume the initial scores at time $$t=0$$ follow an uninformative Gaussian base distribution $$\mathbf{z}_i(0) \sim \mathcal{N}(0,I)$$ for each node $$i$$. Node scores evolve in parallel over time by a differential equation,

<p align="center">
    $$\dot{\mathbf{z}}_{i}(t) := \frac{\partial \mathbf{z}_i(t)}{\partial t} = f_\theta\big( t, \mathbf{z}_i(t), \mathbf{z}_{\mathcal{N}_i}(t),\mathbf{x}_{i}, \mathbf{x}_{\mathcal{N}_i} \big), \qquad i = 1, \ldots, M$$
  </p> 
  
where $$\mathcal{N}_{i} = \{ \mathbf{z}_{j} : (i,j) \in E \}$$ is the set of neighbor scores at time $$t$$, $$\mathbf{x}$$ is the spatial information (2D/3D), and $$\theta$$ are the parameters of the flow function $f$ to be learned. By collecting all node differentials we obtain a **modular** joint, coupled ODE, which is equivalent to a graph PDE [5,6], where the evolution of each node only depends on its immediate neighbors. 

<p align="center">
 $$\dot{\mathbf{z}}_{i}(t) = \begin{pmatrix} \dot{\mathbf{z}}_{i}(t)_1(t) \\ \vdots \\ \dot{\mathbf{z}}_{i}(t)_M(t) \end{pmatrix} = \begin{pmatrix} f_\theta\big( t, \mathbf{z}_1(t), \mathbf{z}_{\mathcal{N}_1}(t) \big) \\ \vdots \\ f_\theta\big( t, \mathbf{z}_M(t), \mathbf{z}_{\mathcal{N}_M}(t) \big) \end{pmatrix} $$
 </p>
## Equivariant local differential

## Training Objective

We reduce the learning problem to maximizing the score cross-entropy $$\mathrm{E}_{\hat{p}_{\mathrm{data}}(\mathbf{z}(T))}[\log p_\theta(\mathbf{z}(T))]$$, where we turn the observed set of graphs $$\{G_{n}\}$$ into a set of scores $$\{\mathbf{z}_{n}\}$$ by using one-hot encoding 
<p align="center">
$$\mathbf{z}_n (G_n; \epsilon) = (1-\epsilon)~\mathrm{onehot}(G_n) ~+~ \dfrac{\epsilon}{|\mathcal{A_s}|} \textbf{1}_{M(n)} \textbf{1}_{|\mathcal{A_s}|}^{\top}~,$$
</p>
where 

We exploit the non-reversible composition of the argmax and softmax to transition from continous space to discrete graph space, but short-circuit in reverse direction. This indeed allows to keep the forward and backward flows aligned.
<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/tikz_diagram.png" />
</p>
We thus maximize an objective over $N$ training graphs, 
<p align="center">
$$\texttt{argmax}_\theta \qquad \mathcal{L} = \mathcal{E}_{\hat{p}_{\mathrm{data}}(\mathbf{z})} \log p_\theta(\mathbf{z}) \approx \frac{1}{N} \sum_{n=1}^N \log p_T\big( \mathbf{z}(T) = \mathbf{z}_n \big)$$     
</p>   

## Molecule Generation

We generate novel molecules by sampling an initial state $$\mathbf{z}(0) \sim \mathcal{N}(0,I)$$ based on structure, and running the modular flow forward in time until $\mathbf{z}(T)$. This procedure maps a tractable base distribution $$p_0$$ to some more complex distribution $$p_T$$. We follow argmax to pick the most probable label assignment for each node.
<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/workflow_final.png" />
</p>

# Results

<p align="center">
  <img src="https://github.com/yogeshverma1998/Modular-Flows-Differential-Molecular-Generation/blob/main/toy_final.png" />
</p>



# References

[1]
[2]
[3]
[4]
[5]
[6]





