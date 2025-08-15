#import "@preview/polylux:0.4.0": *
#import "@preview/cetz:0.4.1"
#import "@preview/cetz-plot:0.1.2": plot, chart
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
#show math.equation: set text(font: "New Computer Modern Math")
#show heading: set block(below: 2em)

#slide[
  #set page(footer: none)
  #set align(horizon)

  #text(1.5em)[Step-3 is Large yet Affordable:  Model-system Co-design for Cost-effective Decoding]

  StepFun Inc.
  
  #text(0.5em)[Aug 18, 2025]
]


#slide[
  = Focused Problem: Optimizing the decoding cost

  Why?

  - low Model FLOPs Utilization (MFU) compared with training and prefill
  - reasoning model needs longer thinking
  - reduce the cost for RL training

  - large room for optimization

]

#slide[
#cetz.canvas({
  import cetz.draw: *
  
  // Draw first square (larger)
  rect((0, 0), (8, 8), fill: blue.lighten(70%), stroke: blue)
  
  // Add text to first square
  content((4, 4), [Attention - reduce\ KV cache size], anchor: "center")

  // Draw second square (larger)
  rect((9, 0), (17, 8), fill: red.lighten(70%), stroke: red)
  
  // Add text to second square
  content((13, 4), [MoE - reduce\ FFN activation], anchor: "center")
})

- Some work uses #text(weight: "bold")[excessive computational cost] to reduce the KV cache size (e.g., ??)
- Some work overly focuses on reducing the FFN activation without considering #text(weight: "bold")[hardware specifications].

]

#slide[
  #box(
    image("./images/models.png", width: 60%),
  )

]


#slide[
  = Helmholtz Machine
  // RBM is a simple generative model with 2 layers. 
  Helmholtz Machine is a hierarchical full-connected feedback NN.

  RBM can be viewed as a Helmholtz machine with only #text(weight: "bold")[one layer of hidden units and an implicit posterior distribution].
]
  // 尽管我们可以将RBM视为具有一个隐藏层和隐式后验分布的Helmholtz机器，但Helmholtz机器可以具有任意数量的隐藏层。同时，Helmholtz机显式得具有生成模型和识别模型。识别模型也就是前文中提到的encoder。也可以比理解为分布，后验分布。


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



]




#slide[
  = Modern development

  Variational Autoencoders (VAEs)

  Generative Adversarial Networks (GANs)

  Flow-based models

  Diffusion models
]

#slide[
  = Works on LLM

  KV-cache compression via importance evaluation

  Block Attention
]
