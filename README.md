*This website contains information regarding the paper Modular Flows: Differential Molecular Generation.*

> **TL;DR:** We propose generative graph normalizing flow models, based on a system of coupled node ODEs, that repeatedly reconcile locally toward globally aligned densities for high quality molecular generation

# Problem of Molecular Generation




# Continuous Normalizing Flows
Normalizing flow have seen widespread use for density modeling, generative modeling , etc which provides a general way of constructing flexible probability distributions. It is defined by a parameterized invertible deterministic transformation from a base distribution $\mathcal{Z}$ (e.g., Gaussian distribution) to real-world observational space $X$ (e.g. images and speech). The $f: \mathcal{Z} \mapsto X$ is an invertible transformation, then we can compute the density function of real-world data $\mathbf{x}$, i.e., $p_X(\mathbf{x})$, via the change-of-variables formula:
<p align="center">
$p_X(\mathbf{x}) = p_{\mathcal{Z}}\big(f_{\theta}^{-1}(\mathbf{x}) \big) \left| \det \frac{\partial f_{\theta}^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|$
 </p>   
Given a datapoint $\mathbf{x}$, the exact density $p_X(\mathbf{x})$ can be computed via inverting the flow by function $f$, $\mathbf{z} = f^{-1}(\mathbf{x})$. Moreover, $\mathbf{x}$ can be sampled from $p_X(\mathbf{x})$ by first sampling $\mathbf{z} \sim p_\mathcal{Z}(\mathbf{z})$ and then performing the feedforward transformation $\mathbf{x} = f_{\theta}(\mathbf{z})$. 

<p align="center">
   ![title](nf_website.png)
</p> 


\citet{ODE} defines a continous analog of Equation~\eqref{eq:1} which replaces the warping function with an integral of continous-time dynamics. 
%The process starts by sampling from a base distribution $\u_0 \sim p_0(\u_0)$. Then, we solve the initial value problem $\u(t_0) = \u_0$, $\du(t) = \frac{\partial \u(t)}{\partial t} = f(\u(t),t;\theta)$, where ODE is defined by the parametric function $f(\u(t),t;\theta)$ to obtain $\u(t_1)$ which constitutes our observable data. These models are called Continous Normalizing Flows (CNF). Then, using the $\emph{instantaneous change of variables}$ formula change in log-density under this model is given as:
%\begin{align}
%    \frac{\partial \log p_t(\u(t))}{\partial t} &= -\tr\left( %\frac{\partial f}{\partial \u(t)} \right)
%\end{align}
%Given a datapoint $\x$, we can compute both the point $\u_{0}$ which generates $\x$, as well as $\log p_1(\x)$ by solving the initial value problem which integrates the combined dynamics of $\u(t)$ and the log-density of the sample resulting in the computation of $\log p_1(\x)$.




# Modular Flows


# Results






