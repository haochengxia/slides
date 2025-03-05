#import "@preview/polylux:0.4.0": *
#let argmax = math.op("arg max", limits: true)
#let argmin = math.op("arg min", limits: true)
#set page(
  paper: "presentation-16-9",
  footer: align(right, text(size: .8em, toolbox.slide-number)),
  margin: (bottom: 2em, rest: 1em),
)
#set text(
  font: "Lato",
  size: 23pt,
)
#show math.equation: set text(font: "Lete Sans Math")
#show heading: set block(below: 2em)

#slide[
  #set page(footer: none)
  #set align(horizon)

  #text(1.5em)[The Helmholtz Machine]

  Peter Dayan, Geoffrey E. Hinton, Radford M. Neal, Richard S. Zemel
  
  #text(0.5em)[February 26, 2025]
]

// #slide[
//   = My first slide

//   Here come my three favourite fonts:
//   #show: later

//   + Atkinson Hyperlegible
//   + Alegreya
//   + TeX Gyre Pagella

//   #show: later

//   And now some math:
//   $
//     sum_(k = 1)^n k = (n (n + 1)) / 2
//   $
// ]

// #slide[
//   = Second slide

//   #toolbox.side-by-side[
//     #rect(width: 100%, height: 1fr)[(imagine this being an image)]
//   ][
//     On the left, you see a #only(2)[not so] beautiful image.
//   ]
// ]

#slide[
  = Intuition
  From (Restricted) Boltzmann machine to Helmholtz Machine

Boltzmann machine's energy function (a column vector {0, 1})
  $
    p(bold(x)) = exp(-E(bold(x))) / Z
  $

  $
    E(bold(x)) = - bold(x)^T U bold(x) - b^T bold(x) =  - sum_(i,j) w_(i,j) x_i x_j - sum_(i) w_i x_i
  $

  Input can be divided into two parts:
  $
    bold(x) = (bold(v), bold(h))
  $
]


#slide[
  = Restricted Boltzmann Machine
  visible variables and hidden variables are fully connected

  #box(
    image("./images/rbm.png", width: 60%),
  )
]

#slide[
  = Restricted Boltzmann Machine
  $
    p(bold(v), bold(h)) = exp(-E(bold(v), bold(h))) / Z
  $

  $
    E(bold(v), bold(h)) = - bold(v)^T W bold(h) - b^T bold(v) - c^T bold(h)
  $

  Then we can get the conditional distribution for hidden variables given visible variables (encoder):
  $
    p(bold(h) | bold(v)) = product_(j=1)^n_h sigma((2 bold(h) - 1) * (c + W^T bold(v)))_j
  $
]

#slide[
  = Restricted Boltzmann Machine
  Decoder:
  $
    p(bold(v) | bold(h)) = product_(i=1)^n_v sigma((2 bold(v) - 1) * (b + W^T bold(h)))_i
  $

  Learning with Contrastive Divergence.
]

#slide[
  = Helmholtz Machine
  // RBM is a simple generative model with 2 layers. 
  Helmholtz Machine is a hierarchical full-connected feedback NN.

  RBM can be viewed as a Helmholtz machine with only #text(weight: "bold")[one layer of hidden units and an implicit posterior distribution].
]
  // 尽管我们可以将RBM视为具有一个隐藏层和隐式后验分布的Helmholtz机器，但Helmholtz机器可以具有任意数量的隐藏层。同时，Helmholtz机显式得具有生成模型和识别模型。识别模型也就是前文中提到的encoder。也可以比理解为分布，后验分布。

#slide[
  = Helmholtz Machine
    #box(
    image("images/RBM_back.png", width: 40%),
  )
  #box(
    image("images/helmholtz-machine.jpg", width: 40%),
  )
]

#slide[
  = Divide generative model and recognition model

  Why it is called Helmholtz Machine? Two models are trained iteratively.

  MLE, theta is model parameters:
  $
    theta = argmax(theta) sum_(i=1)^n log p(bold(v)_i | theta)
  $

#show strong: set text(red)
  $
    theta = argmax(theta) sum_(i=1)^n log p(bold(v)_i, bold(h) | theta)
  $

  // we have a large space for the hidden variables. it makes the optimization problem more difficult.
]
#slide[
  = Real posterior distribution is intractable
  $
    p(bold(h) | bold(v), theta) = p(bold(h), bold(v) | theta) / p(bold(v) | theta)
  $

  Approximation with Q:
  $
    Q(bold(h) | bold(v)) = product_(i=1)^Q (h_i | v)
  $

  ELBO (Evidence Lower Bound) from variational inference:
  $
    log p(bold(v) | theta)  = E_Q [log p(bold(v), bold(h) | theta) - log Q(bold(h) | bold(v))] + D_(K L)(Q(bold(h) | bold(v)) || p(bold(h) | bold(v), theta))
  $



  // 实线表示由参数 $φ$ 表示的自下而上的识别过程，虚线表示由参数 $θ$ 表示的自上而下的生成过程。每个神经元的活动是根据其前一层所有神经元的活动计算得出的。激活函数如下所示。

  // $
  // q_i^l(phi, s^(l-1)) = σ( sum_(k) φ_(k,i)^(l-1,l) × s_k^(l-1))
  // $

  // $
  // p_i^l(θ, s^(l+1)) = σ( sum_(k) θ_(k,i)^(l+1,l) × s_k^(l+1))
  // $

  // 其中 $s$ 表示神经元活动，$s ∈ {0,1}$。上标 $l$ 表示层索引，下标表示该层的特定神经元。
]

#slide[
  = Real posterior distribution is intractable
  $
    D_(K L)(Q(bold(h) | bold(v)) || p(bold(h) | bold(v), theta)) > 0
  $

  Therefore, the ELBO is a lower bound of the log-likelihood.

  $
    L(theta, Q) = E_Q [log p(bold(v), bold(h) | theta) - log Q(bold(h) | bold(v))]
  $

  That is, $log p(bold(v) | theta) >= L(theta, Q)$.


]

#slide[
  = Introduce free energy
  $
    F(bold(v), theta) = - E_Q [log p(bold(h), bold(v) | theta)] - H(Q(bold(h) | bold(v)))
  $

  The objective is to minimize the free energy.

  Now we have two models:
  $p(bold(v), bold(h) | theta)$
  and
  $Q(bold(h) | bold(v); phi)$

  We can use the free energy to optimize the parameters.

  $
    theta = argmin(theta) F(bold(v), theta)
  $

  $
    phi = argmin(phi) F(bold(v), theta)
  $

]

#slide[
  = Bi-level optimization
  Wake-sleep algorithm
  The wake phase is to minimize the free energy, and the sleep phase is to maximize the free energy. 

#show strong: set text(green)
  *Wake* phase: target is minimize $F(bold(v), theta)$, i.e., maximize $L(theta, Q)$. use real data pass Q to get posterior distribution. then update theta.
  
#show strong: set text(blue)
  *Sleep* phase: generate data with $p(bold(v), bold(h) | theta)$. pass generated data to Q to get posterior distribution. then update phi.

]


#slide[
  = Modern development

  Variational Autoencoders (VAEs)

  Generative Adversarial Networks (GANs)

  Flow-based models

  Diffusion models
]

#slide[
  = Variational Autoencoders (VAEs)
  #box(
  image("./images/vae.png", width: 80%),
  )
]

#slide[
  = Works on LLM

  KV-cache compression via importance evaluation

  Block Attention
]
