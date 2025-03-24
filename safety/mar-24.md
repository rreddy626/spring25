# Safety - Adversarial Robustness


## Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope
*[Eric Wong, J. Zico Kolter](https://arxiv.org/abs/1711.00851)*

  

## **Introduction and Motivation**

Adversarial examples, inputs perturbed by imperceptible adversarial noise to mislead machine learning models, expose a critical vulnerability in neural networks. While empirical defenses like adversarial training (Madry et al., 2018) improve robustness, they lack formal guarantees, leaving models susceptible to adaptive attacks. This paper addresses the challenge of certifiable robustness: mathematically proving that a model’s predictions remain unchanged for all perturbations within a bounded threat model (e.g., $\ell_\infty$-ball of radius $\epsilon$).

  

### **Core Motivations**:

1. **Safety-Critical Applications**: Autonomous systems (e.g., self-driving cars, medical diagnostics) require guarantees that models will not fail under adversarial inputs.

2. **Limitations of Heuristic Defenses**: Adversarial training alone cannot certify robustness; certified methods like interval bound propagation (IBP) (Gowal et al., 2018) are overly conservative.

3. **Bridging Theory and Practice**: Develop a framework that balances computational tractability with non-conservative robustness bounds.

  

The authors propose constructing a **convex outer adversarial polytope**—a convex relaxation of the set of network outputs under adversarial perturbations—to derive provable robustness certificates via linear programming (LP).

  

---

  

## **Methods**

The method revolves around bounding the output of a neural network under adversarial perturbations by relaxing non-convex activation functions into linear constraints. Key steps include:

  

### **1. Adversarial Polytope Definition**

For an input $x$, perturbation budget $\epsilon$, and network $f$, the adversarial polytope $\mathcal{Z}_\epsilon(x)$ is defined as:

$$
Z_\epsilon(x) = \lbrace f(x + \delta) \mid \|\delta\|_\infty \leq \epsilon \rbrace.
$$



Certifying robustness requires verifying that all points in $Z_\epsilon(x)$ map to the same class. Directly analyzing $Z_\epsilon(x)$ is intractable due to non-convexity from ReLU activations.

  

### **2. Convex Relaxation of ReLU Activations**

Each ReLU activation $z_j = \max(0, \hat{z}_j)$ (where $\hat{z}_j = W_j x + b_j$) is relaxed using **linear upper and lower bounds** on $\hat{z}_j$. These bounds are computed via interval arithmetic:

- Let $l_j \leq \hat{z}_j \leq u_j$ be pre-activation bounds.

- The ReLU can be outer-approximated by:

$$
z_j \geq 0, \quad z_j \geq \hat{z}_j, \quad z_j \leq \frac{u_j}{u_j - l_j}(\hat{z}_j - l_j).
$$

This forms a convex envelope around the ReLU’s non-convex output (see Figure 1 in the paper).

  

### **3. Linear Programming for Robustness Certification**

The relaxed adversarial polytope is encoded as an LP to compute the worst-case output deviation. For a network with $L$ layers and input $x$, the certification LP is:

$$
\begin{align*}
\text{Maximize} \quad & c^T z_L \quad \text{(e.g., $c = e_y - e_{y'}$ for target class $y'$)} \\
\text{Subject to} \quad & z_{k+1} \geq W_k z_k + b_k \quad \text{(pre-ReLU)} \\
& z_{k+1} \geq 0 \\
& z_{k+1} \leq \frac{u^{(k)}}{u^{(k)} - l^{(k)}}(W_k z_k + b_k - l^{(k)}) \\
& \|z_0 - x\|_\infty \leq \epsilon \quad \text{(input constraint)},
\end{align*}
$$

where $l^{(k)}$ and $u^{(k)}$ are pre-computed bounds for layer $k$. Solving this LP yields the maximum possible deviation $c^T z_L$; if this value is negative, the input is certifiably robust.

  

### **4. Dual Optimization for Training Robust Networks**

To train networks that are both accurate and certifiably robust, the authors propose a **dual optimization** approach:

1. **Robust Loss Term**: Penalize the worst-case margin between the correct class $y$ and the most adversarial class $y'$. For each sample $(x, y)$, compute:

$$
L_{\text{robust}}(x, y) = \max_{y' \neq y} \left( \text{LP-solver}(x, y, y') \right),
$$

where $\text{LP-solver}$ returns the maximum deviation $c^T z_L$.

2. **Combined Objective**: The total loss is a weighted sum of cross-entropy and robustness loss:

$$
L(\theta) = (1-\lambda) L_{\text{ce}}(f(x), y) + \lambda L_{\text{robust}}(x, y).
$$

Here, $\lambda$ balances accuracy and robustness.

  

### **5. Scalability via Bound Propagation**

To avoid solving full LPs during training, the authors use **bound propagation** (e.g., CROWN (Zhang et al., 2018)) to compute layer-wise bounds $l^{(k)}$ and $u^{(k)}$ efficiently. For a layer $k$ with weights $W_k$ and input bounds $[l^{(k-1)}, u^{(k-1)}]$, the pre-activation bounds are:

$$
l^{(k)} = W_k^+ l^{(k-1)} + W_k^- u^{(k-1)} + b_k, \\
u^{(k)} = W_k^+ u^{(k-1)} + W_k^- l^{(k-1)} + b_k,
$$

where $W_k^+ = \max(W_k, 0)$ and $W_k^- = \min(W_k, 0)$. This enables fast computation of bounds without solving LPs.

  

---

  

## **Key Findings**

- **Certifiable Robustness**

	- Achieves 70.3% certified robust accuracy on MNIST ($\epsilon=0.1$) and 30.2% on CIFAR-10 ($\epsilon=2/255$), outperforming IBP and PGD-based adversarial training.

	- Certificates hold for all perturbations within the $\ell_\infty$-ball, providing formal guarantees.

  

- **Tighter Relaxations**

	- The convex outer polytope produces less conservative bounds than interval propagation (e.g., 20% improvement on CIFAR-10).

  

- **Efficiency**

	- LP-based certification scales to networks with about 100,000 parameters (e.g., 4-layer CNNs).

	- Training with bound propagation reduces computational overhead by 10× compared to full LP-based training.

  

- **Theoretical Contributions**

	- Introduces the first framework for joint training and certification of ReLU networks.

	- Extends to other activation functions (e.g., sigmoid) via analogous relaxations.

  

---

  

## **Critical Analysis**

### **Strengths**

- **Theoretical Soundness**
Provides the first scalable method for certifying ReLU networks against $\ell_\infty$ attacks with non-conservative bounds.

- **Unified Training and Certification**
By integrating robustness penalties into the loss function, the method avoids the "certification gap" seen in post-hoc methods.

- **Flexibility**
The framework generalizes to broader threat models (e.g., $\ell_1$, $\ell_2$) by modifying the input constraint in the LP.

  

### **Limitations**

- **Scalability**

	- LP complexity grows cubically with network width, limiting applicability to large architectures (e.g., ResNet-152).

	- Bound propagation introduces approximation errors that compound with network depth.

  

- **Conservatism**

	- The convex relaxation overestimates the adversarial polytope, leading to false negatives (missed certifications). For example, on CIFAR-10, nearly 15% of truly robust samples are not certified.

  

- **Threat Model Restrictions**

	- Focuses on $\ell_\infty$ perturbations; extensions to other norms (e.g., $\ell_1$) require rederiving bounds and may increase computational cost.

  

- **Accuracy-Robustness Trade-Off**

	- Even with optimal $\lambda$, standard accuracy drops by about 10% on CIFAR-10 compared to non-robust models.

  

### **Future Directions**

- **Tighter Relaxations**

	- Use semi-definite programming (SDP) or mixed-integer linear programming (MILP) for non-ReLU activations.

	- Explore adaptive relaxations that tighten bounds selectively (e.g., near decision boundaries).

  

- **Scalability Improvements**
	
	- Develop GPU-accelerated LP solvers or exploit sparsity in network weights.

	- Integrate with stochastic training methods (e.g., SGD with bound propagation).

  

- **Broader Applications**

	- Extend to semantic perturbations (e.g., rotations, translations) and non-vision tasks (e.g., NLP, reinforcement learning).

  

- **Hybrid Defenses**

	- Combine convex relaxation with empirical defenses (e.g., randomized smoothing) for improved robustness.

  

---

  

## **Conclusion**

This paper proposed a framework for certifying neural network robustness via convex relaxations of the adversarial polytope. By reformulating robustness verification as an LP and integrating it into training, the method bridges the gap between empirical and provable defenses. While scalability and conservatism remain challenges, the work lays a foundation for future research in certifiable adversarial robustness, with implications for safety-critical AI systems. The convex outer polytope approach has already inspired follow-up work (e.g., α-CROWN, MN-BaB), highlighting its enduring impact on the field.

---
# Scaling Provable Adversarial Defenses

*[Eric Wong, Frank R. Schmidt, Jan Hendrik Metzen, J. Zico Kolter](https://arxiv.org/pdf/1805.12514)*

---

## Introduction and Motivation
In recent years, adversarial attacks have revealed significant vulnerabilities in deep neural networks. Adversarial attacks are small, crafted  perturbations to inputs that cause models to misclassify objects while the changes remain nearly invisible to humans. As these attacks become more sophisticated, ensuring robustness against them has become crucial for deploying reliable machine learning systems. To avoid this, machine learning models try to focus on having adversarial robustness, the ability of a machine learning model to maintain its performance when faced with adversarial inputs. Recently it has been theorized that there are formal guarantees of robustness against adversarial attacks, provable robustness, backed by mathematical proofs that ensure they will remain robust under certain conditions. Existing methods only provide provable robustness for small networks, limiting their practical use in complex models. This paper focuses on developing methods to create deep neural networks that are provably robust against adversarial attacks.

This paper tackles the challenge of scaling provable defenses to deeper, more complex architectures. Building on the work of Wong and Kolter (2017), the authors introduce three key innovations to improve scalability and robustness. First, they extend provable robustness techniques to handle networks with skip connections, residual layers, and general activation functions, making the approach compatible with modern architectures. Second, they address computational complexity by introducing a nonlinear random projection technique that reduces complexity from quadratic to linear in the number of hidden units, making robust training more scalable. Finally, they propose the use of cascade models, a method of training multiple classifiers in stages, where each stage handles the examples the previous stage cannot robustly classify.

--- 
## Methods

### Provable Robustness for General Deep Learning Networks

This paper seeks to approach handling skip networks and arbitrary activation functions. Earlier methods only worked for simple feedforward networks with linear layers followed by activations.

**Adversarial Problem Formulation:** The goal of this formula to make the network robust to perturbations.
	Let $f_{\theta} : R^{|x|} \rightarrow R^{|y|}$ represent a k-layer neural network, defined as:
 
$z_i \sum_{j=1}^{i-1} f_{ij}(z_j)$, for $i = 2,...,k$


Where:

- $z_1 = x$ is the input.    
- $f_{ij}(z_j)$ represents the transformation from layer $j$ to layer $i$, encompassing operations like linear transformations, skip connections, and activations.    
- $f_{\theta}(x) = z_k$ is the output.


The adversarial perturbation set is the collection of all possible inputs that result from adding a small, controlled disturbance $\delta$ to the original input (x) constrained by a maximum perturbation size $\epsilon$: $B(x) = {x + \delta : ||\delta|| \leq \epsilon}$. This set defines the "space" an adversary can explore to try and fool the model and push it toward incorrect predictions while staying within a certain limit.

The adversary aims to find the worst case perturbation that minimizes the model's confidence in the correct output defined by:

$min_{z_k} c^Tz_k$,  *subject to* $z_i \sum_{j=1}^{i-1} f_{ij}(z_j)$,$ $z_i$  $\epsilon$  $B(x)$

Where c is a vector used to formalize the adversarial objective such as reducing the confidence in the true class while increasing it for the target class.

**Fenchel Duality for Lower Bounding:** The paper uses Fenchel Duality to reformulate the adversarial problem as a dual optimization task. Each network operation (e.g., linear layers, ReLU, skip connections) has its own dual formulation. This dual formulation provides a provable lower bound on the adversarial loss, simplifying robustness analysis.

Fiven a function f(x), its Fenchel conjugate f*(y) is defined as: $f^*(y) = sup_x(x^Ty-f(x))$.

Using this formulation, the adversarial problem can be lower-bounded as: $J(x,\nu_{1:k}) = -\nu_1^Tx - \epsilon ||\nu_1||^* - \sum_{i=1}^{k=1} h_i (\nu_{i:k})$

Where:
- $\nu_{i:k}$ : Dual variables representing each layer in the network.
- $||\nu_1||^*$ : Dual norm of the perturbation bound.
- $h_i(v_{i:k})$: Upper bound on the conjugate function for each layer i

By applying Fenchel Duality to each operation, the dual of the entire network is constructed by combining the modular components. This modular approach enable the construction of dual networks automatically, and simplifies the problem by turning adversarial robustness into a layer based analysis of dual forms, making it feasible for complex architectures like ResNets.

### Efficient Bound Computation with Random Projections
In ReLU networks with $\ell_\infty$ -bounded perturbations, earlier methods computed the upper bound on the robust loss by calculating contributions from every hidden unit, leading to **quadratic complexity** in the number of hidden units. Specifically, computing the $\ell_1$ norm (dual of $\ell_\infty$ norm) requires processing of each unit individually, which is costly. As a solution, the paper introduces nonlinear random projections to approximate these $\ell_1$ related terms, reducing complexity from quadratic to **linear**.

Cauchy random projections are utilized because they are heavy-tailed making them ideal for approximating the $\ell_1$ norm through random projections.

Let $\nu_1$ represent the dual network's first layer, and *R* by a standard Cauchy random matrix. The $\ell_1$ norm is approximated as:

$||\nu_1||_1 \approx  median(|\nu_1^T R|)$

For terms involving ReLU activations and perturbations, the paper approximates sums over hidden units as: 

$\sum_{j \epsilon I}\ell_{ij}[v_{ij}]^+ \approx \frac{1}{2} (-\text{median} (|v_i^T \text{diag} (d_i)R|) + \nu_i^Td_i)$

Random projections eliminate the need to explicity pass each hideen unit through the dual network, saving computation time. This computation reduces complexity while maintaining the tightness of the robustness bounds.

### Cascading Models

Robust training can over-regularize a network which leads to underfitting of the model. A single robust classifier would be responsible for handling both examples that are deemed easier and harder, leading to a trade off between robustness and accuracy. Instead of relying on a single robust classifier, the cascade approach trains multiple classifiers sequentially. Each classifier focuses on examples that the previous classifiers failed to certify as robust. This progressively improves robustness while distributing the workload across multiple smaller models.

**Cascade Training Procedure**:
1. Train the first model on the full dataset (f1)
2. Filter Certified Examples
3. Train the next model (f2 only on the uncertified examples of f1)
4. Continue this process training more models refining robust predictions

The cascade procedure is an improvement as earlier models in the cascade handle easy to certify examples, leaving more challenging examples for later models to specialize in. It allows for the use of smaller more efficient models without sacrificing robustness. In turn, this reduces the burden on each individual model and verified robust error significantly compared to single robust classifier.

### Testing and Experimentation

The authors methods were evaluated on two datasets:
- MNIST: Digit classification with 28x28 grayscale images
- CIFAR-10: 10-class color image classification

Tested on different architectures:
- Small network: Two convolutional layers (16 and 32 filters) and one fully connected layer (100 units)
- Large network: Four convolutional layers (32, 32, 64, 64 filters) and two fully connected layers (512 units)
- Four residual blocks with increasing filter sizes (16, 16, 32, and 64 filters)

Results:
- On MNIST, the robust error was reduced from 5.8% to 3.1% on best cascade for $\epsilon = 0.1$.
- On CIFAR-10, robust error dropped from 80% to 36.4% on best cascade for $\epsilon = \frac{2}{255}$.
- Cascade models consistently improved robust performance, but at the cost of slightly increased non robust error.
- The use of random projections showed negligible impact on accuracy while drastically reducing computation time, making robust training feasible for larger models.

--- 
## Critical Analysis

### Strengths and Key Findings

- **Extending Robust Training to Larger Networks**
	- Previous work applied to simple feedforward networks while this paper generalizes the approach to more complex architectures including those with skip connections and non linear movement.
	- Utilizes a modular technique to apply methods automatically to any network structure.

- **Improved Computational Efficiency**
	- Proposes a nonlinear random projection technique that reduces the complexity to linear, making robust training much more scalable.

- **Cascade Models**
	- Introduces cascade models to improve robustness which involves training multiple stages of classifiers where each stage handles the examples the previous stage could not classify robustly. 

### Weaknesses
- Experiments focused only on MNIST and CIFAR-10 and evaluation is limited to convolutional networks and ResNets, possibly overlooking performance variations in other architectures.
- The paper primarily addresses $\ell_\infty$-bounded perturbations, leaving other models like $\ell_2$ or spatial transformations underexplored.
- Cascade models improve robustness but at the expense of increased non robust error, potentially reducing practical utility.
- The use of random projections introduces approximations that, while efficient, may loosen the robustness guarantees.

### Potential Biases and Ethical Considerations

- Robustness improvements might not be uniformly effective across different data subgroups, potentially introducing bias if not carefully evaluated across diverse datasets.

---

## Towards Deep Learning Models Resistant to Adversarial Attacks
*[Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu](https://arxiv.org/abs/1706.06083)*

## Introduction and Motivation

Adversarial inputs are when small changes to an input image deceives neural networks with high confidence, resulting in misclassification. For instance, if a neural network is given the image of a stop sign which has changes that are imperceptible to a human (i.e. changes to the image that humans cannot see), neural networks may misclassify the image as a yield sign instead of a stop sign, which can have disastrous effects in the field of autonomous driving.

This paper seeks to answer the following question:

*How can we train deep neural networks that are robust to adversarial inputs?* 

This paper aims to develop neural networks with adversarial robustness through the application of robust optimization techniques.

## Methods

To develop neural networks with adversarial robustness, the authors use the natural saddle point (min-max) formulation:

$min_{\theta} \rho(\theta), \quad \text{where} \quad \rho(\theta) = E_{(x,y) \sim D} \left[ \max_{\delta \in S} L(\theta, x + \delta, y) \right]$

The saddle point problem consists of an inner maximization problem and an outer minimization problem.

The inner maximization problem aims to find an adversarial version of a given data point *x* that achieves a high loss.

The outer minimization problem aims to find model parameters so the "adversarial loss" given by the inner attack problem is minimized. This essentially trains a classifier to be robust using adversarial training techniques.

Through the combination of finding the adversarial version with the greatest loss and trying to minimize that loss by finding the correct model parameters, the saddle point problem allows for resistance to a broad class of attacks rather than defending against only specifically known attacks. Adversarial training directly corresponds to optimizing the saddle point problem, which allows for casting both attacks and defenses into a common theoretical framework.

### Inner Maximization Problem

To train a neural network to be adversarially robust, the network must be trained on strong adversarial examples.

*How can we produce strong adversarial examples, i.e., adversarial examples that fool a model with high confidence while requiring only a small perturbation?*

This is found by using projected gradient descent (PGD), which was found to be a "universal" adversary among first-order approaches:

$$
x_{t+1} = \Pi_{x + S} \Big( x^t + \alpha \cdot \text{sgn}(\nabla_x L(\theta, x, y)) \Big)
$$

PGD iteratively modifies the image using the gradient of the loss function to maximize the model's error while ensuring the perturbation remains small. If a modification exceeds the allowed perturbation limit, it is projected back within the constraint. This process keeps the adversarial example close to the original image while altering it just enough to cause misclassification.


### Outer Minimization Problem

To train a neural network to be adversarially robust, the network must be trained to classify adversarial examples correctly, such that there is minimal loss.

*How can we train a model so that there are no adversarial examples, or at least so that an adversary cannot find them easily?*

The goal of the outer minimization problem is to ensure that no adversarial perturbations are possible by maintaining a consistently low loss across all perturbations.

### Experiments

The experiments were conducted using two datasets: MNIST and CIFAR-10. MNIST consists of labeled images of handwritten digits, while CIFAR-10 contains labeled images categorized into ten distinct classes. In all the experiments, the adversary of choice is PGD.

One of the experiments found that by training against an adversarial example in both datasets, the training loss decreased, showing that it is possible to reduce the value of the inner problem of the saddle point formulation.

The trained models, using data from MNIST and CIFAR-10, were evaluated against various adversarial attacks. For MNIST, the adversaries included Natural (no attack), FGSM, PGD, Targeted, CW, and CW+. The source networks for these attacks were the model itself (white-box attack), an independently initialized and trained copy of the network, and a different architecture. The evaluation results on MNIST showed classification accuracy ranging from 89.3% to 98.8%, demonstrating strong robustness against adversarial attacks.

Models trained on the CIFAR10 dataset were evaluated against the following adversarial attacks: Natural, FGSM, PGD, and CW. The attacks were sourced from three types of networks: the model itself (white-box attack), an independently initialized and trained copy of the network, and a version trained solely on natural examples. The adversarially trained network achieved classification accuracy between 45.8% and 87.3%, suggesting a degree of robustness to adversarial inputs but also highlighting potential areas for improvement.

The previous experiments used an ϵ value of 0.3 for MNIST and an ϵ value of 8 for CIFAR10, where ϵ represents the perturbation bound - the maximum permissible modification to an input when generating adversarial examples. This limits the extent to which an adversary can alter each pixel while ensuring the changes remain imperceptible to humans. During training, the authors observed that for a smaller ϵ than the one used during training, the models achieve equal or higher accuracy.

## Key Findings

- **Deep Neural Networks Can Be Made Resistant to Adversarial Attacks**

	As demonstrated in the paper, this process can be formulated as a saddle point optimization problem by training networks on adversarial inputs. First, the adversarial input that maximizes the loss is identified. Then, the model parameters are adjusted to minimize that loss, improving robustness against adversarial attacks.

- **Increasing the Capacity of the Network When Training Increases Robustness Against Perturbations**

	When the network has a larger capacity, it can create more complicated decision boundaries to correctly classify natural and adversarial examples. When the network has limited capacity, it prioritizes adversarial robustness at the cost of performance on natural examples, leading to a trade-off between accuracy and resilience.

- **Projected Gradient Descent (PGD) is the Strongest First-Order Adversary**

	PGD is the strongest attack that utilizes local first-order information about the network.

## Critical Analysis

### Strengths
- **Mathematical Formulation for Adversarially Robust Training** 
The paper provides a mathematically rigorous analysis of training neural networks to be robust against adversarial inputs by using the saddle point formula (min-max optimization problem).

- **Clarifies Resources Needed for Networks to be Adversarially Robust**
The paper addresses how networks need a large capacity to successfully create the complex decision boundaries needed for adversarial robustness.

- **Demonstrates the Effectiveness of PGD Adversarial Training** 
The paper demonstrates that PGD is a strong first-order adversary, and training models against PGD attacks leads to effective generalization for adversarial robustness.

### Weaknesses

- **Models Trained on CIFAR10 Did Not Achieve High Accuracy Compared to MNIST**
Models trained on the MNIST dataset achieved significantly higher accuracy compared to those trained on CIFAR10, indicating that the models on CIFAR10 did not attain the same level of robustness as those on MNIST.

- **Computationally Expensive**
Achieving a high level of robustness requires a network with a large capacity, which can be computationally expensive.

### Potential Biases

- **Limited Datasets Evaluated**
Experiments were conducted solely on the MNIST and CIFAR10 datasets. Analyzing additional datasets is necessary to determine whether the saddle point formulation is effective universally or if it is specific to MNIST.

### Ethical Considerations

The paper raises questions about how training neural networks to be robust against adversarial inputs can affect how natural (unchanged) data is classified - there may be a tradeoff between robustness and accuracy. Adversarial robustness is especially concerning in real world security applications, such as in autonomous vehicles and biometric classification.

---
## Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks
*[Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami](https://arxiv.org/abs/1511.04508)*

  

## Introduction and Motivation

  

Deep learning has achieved remarkable success in various machine learning tasks, particularly in input classification using Deep Neural Networks (DNNs). However, like other machine learning techniques, DNNs are vulnerable to adversarial samples – carefully crafted inputs designed to force the network to produce incorrect, adversary-selected outputs. These attacks pose significant security risks in critical applications such as autonomous vehicles, content filtering, and biometric authentication. The paper introduces a novel defense mechanism called "defensive distillation" to mitigate the effectiveness of adversarial samples on DNNs. The motivation behind this work is to address the growing security concerns surrounding the deployment of DNNs in sensitive domains by developing a training procedure that enhances their robustness against malicious input perturbations.

  

## Methods

  

The paper proposes "defensive distillation," a modified training procedure based on the concept of knowledge distillation. Instead of transferring knowledge between different DNN architectures (as in traditional distillation for model compression), defensive distillation uses the knowledge extracted from a DNN to improve its own resilience to adversarial samples.

  

The method involves the following steps:

  

1. **Training a Teacher Network with High Temperature**
A DNN is initially trained on the original dataset using a softmax output layer with a high temperature (T > 1). This "teacher" network produces "soft labels," which are probability distributions over the classes, containing more information about the model's confidence and the relationships between classes than traditional "hard labels" (the single correct class). The softmax function with temperature T is given by:

$$F_i(X) = \frac{e^{z_i(X)/T}}{\sum_{l=0}^{N-1} e^{z_l(X)/T}}$$

where $z_i(X)$ are the logits (outputs of the last hidden layer) for class $i$.

  

2. **Training a Student Network with High Temperature on Soft Labels**
A second DNN (which can be the same architecture as the teacher) is trained on the same dataset, but using the soft labels generated by the teacher network as the target outputs. This "student" network is also trained using a softmax layer with the same high temperature T.

  

3. **Deployment with Temperature Set to 1**
At test time, the trained student network is used for classification with the softmax temperature set back to 1 to produce more discrete and confident predictions.

  

The authors empirically evaluated this defense mechanism on two 9-layer convolutional neural network architectures trained on the MNIST and CIFAR10 datasets. They used a specific adversarial attack method [7] that computes the Jacobian of the network output with respect to the input and uses a saliency map to select and perturb a limited number of input features to achieve source-target misclassification. The success rate of this attack on models trained with and without defensive distillation was compared, and the impact of varying the distillation temperature was also investigated. Additionally, the paper analyzed the effect of distillation on the amplitude of adversarial gradients and the robustness of the DNNs (measured by the minimum perturbation required for misclassification).

  

## Key Findings

  

The study yielded several significant findings:

  

* **Reduced Adversarial Success Rate**
Defensive distillation dramatically reduced the success rate of adversarial sample crafting. On the MNIST dataset, the success rate decreased from 95.89% for the original model to less than 0.5% for the distilled model (at a temperature of T=100). Similarly, for CIFAR10, the success rate dropped from 87.89% to around 5% (at T=100).

* **Impact of Distillation Temperature**
Increasing the distillation temperature generally made adversarial sample crafting harder. There was an observed "elbow point" in the temperature-success rate curve, suggesting an optimal temperature range for defense.

* **Maintained Classification Accuracy**
The use of defensive distillation did not significantly degrade the classification accuracy on clean samples. The accuracy variability between models trained with and without distillation was less than 1.37% for both MNIST and CIFAR10, and in some cases, a slight improvement was observed.

* **Reduced Gradient Sensitivity**
Defensive distillation led to a substantial reduction in the amplitude of adversarial gradients used for crafting malicious inputs, by a factor of up to $10^{30}$. This smoothing effect makes the network less sensitive to small input perturbations.

* **Increased Robustness**
Distillation increased the average minimum number of features that needed to be modified to create adversarial samples. For the MNIST model, this increased by about 790%, and for CIFAR10 by about 556%, indicating enhanced robustness against small perturbations.

* **Increased Confidence**
Distillation was also found to increase the confidence of the distilled model's predictions on the CIFAR10 dataset.

  

## Critical Analysis

  

### Strengths

  

* **Novel Defense Mechanism**
The paper introduces a creative and effective defense mechanism against adversarial samples by repurposing the distillation training procedure.

* **Significant Empirical Results**
The empirical evaluation on two standard datasets (MNIST and CIFAR10) demonstrates a substantial reduction in the effectiveness of a strong adversarial attack. The quantitative results are compelling.

* **Analytical Justification**
The paper provides an analytical explanation for why distillation leads to increased robustness, linking it to the smoothing of the learned classifier function and the reduction of gradient magnitudes.

* **Comprehensive Evaluation**
The study explores the impact of a key hyperparameter (distillation temperature) on both adversarial robustness and classification accuracy, providing valuable insights for practical deployment.

* **Clear Problem Definition and Motivation**
The paper clearly articulates the problem of adversarial vulnerabilities in DNNs and effectively motivates the need for robust defense mechanisms.

  

### Weaknesses

  

* **Specific Attack Method Evaluated**
The study primarily focuses on the Jacobian-based saliency map attack [7]. While this is a well-known attack, the defense's effectiveness against other types of adversarial attacks (e.g., gradient-based attacks like FGSM or PGD, or black-box attacks) is not extensively explored in this excerpt. The paper mentions discussing the impact with other crafting algorithms in Section VI (not fully provided).

* **Temperature Sensitivity**
The effectiveness of defensive distillation is dependent on the choice of the distillation temperature. Finding the optimal temperature that balances robustness and accuracy might require careful tuning for different datasets and architectures.

* **Computational Cost**
While the paper highlights that distillation can be used for model compression in its traditional form, the defensive distillation approach involves training at least one extra model (the teacher), which could increase the overall training time.

* **Limited Scope of Model Architectures**
The evaluation is limited to specific 9-layer convolutional neural networks. The generalizability of these findings to other types of DNN architectures (e.g., recurrent neural networks, transformers) or significantly deeper networks is not explicitly demonstrated.

* **Distance Metric for Robustness**
The robustness metric used in the paper relies on the number of perturbed features. While this is a common metric, other distance metrics (e.g., L2 norm of the perturbation) might provide a more comprehensive understanding of the perturbation magnitude.

  

### Potential Biases

  

* **Choice of Datasets and Attack**
The evaluation is performed on two image classification datasets. The effectiveness of defensive distillation might vary for other data modalities (e.g., text, audio). Similarly, the choice of the specific adversarial attack method could influence the observed results.

* **Hyperparameter Selection** 
The choice of training hyperparameters (learning rate, number of epochs, etc.) for both the original and distilled models could potentially introduce bias if not carefully optimized. However, the paper states that their DNN performance is consistent with previous work.

  

### Ethical Considerations

  

* **Dual-Use Nature**
While defensive distillation aims to protect DNNs from adversarial attacks, the knowledge gained from understanding adversarial vulnerabilities and developing defenses could potentially be misused to create more sophisticated and evasive attacks. This is a common ethical consideration in the field of security.

* **Accessibility of Defense**
The implementation and deployment of defensive distillation might require specific technical expertise and computational resources, potentially creating a disparity in security capabilities between different organizations or individuals.

* **Impact on Performance**
While the paper shows minimal impact on accuracy, any reduction in the performance of critical systems due to the defense mechanism could have ethical implications, especially in safety-critical applications.

  

In conclusion, the paper presents a promising defense mechanism against adversarial samples in DNNs. Defensive distillation offers a significant improvement in robustness with minimal impact on accuracy. However, further research is needed to evaluate its effectiveness against a wider range of attacks, on different model architectures and data modalities, and to further understand the optimal parameter settings and potential limitations.

---
## Theoretically Principled Trade-off between Robustness and Accuracy
*[Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, Michael I. Jordan](https://arxiv.org/abs/1901.08573)*

## Introduction and Motivation

  

Deep learning models have revolutionized fields from computer vision to natural language processing. However, they also suffer from adversarial attacks. These attacks are often regarding small modifications to the input data that can cause highly confident misclassifications. For example, an image classifier could confidently misidentify a cat as a dog just due to a few pixel changes. The main question discussed in this paper is if we can design a model that is both highly accurate and robust to adversarial attacks. The authors of his paper prove that robustness and accuracy inherently collide with each other – a model that is more robust will have to sacrifice some accuracy, and vice versa. However, the authors turn this limitation into an optimization problem. The paper discusses a new framework TRADES (Tradeoff-inspired Adversarial Defense via Surrogate-loss minimization) – an algorithm that balances accuracy and robustness by providing a new benchmark for adversarial defense.

  

**Key Questions Addressed in This Paper**

1.  **Why do deep learning models struggle with adversarial robustness?**
    Small adversarial modifications can drastically change a model’s predictions, revealing a fundamental weakness in conventional training methods.
    

2.  **What is the theoretical trade-off between robustness and accuracy?**
    The authors mathematically decompose the robust error into natural error and boundary error, proving that perfect robustness cannot be achieved without sacrificing accuracy.
    

3.  **How can we design a defense that explicitly balances robustness and accuracy?**
    Using TRADES, the authors develop a novel adversarial defense that optimizes a carefully designed loss function to control the robustness-accuracy trade-off.
    

  

By developing TRADES, the authors not only advance the theoretical understanding of adversarial robustness, but also provide a practical defense method that achieves high performance across various datasets.

  
  

## Methods

1.  **Understanding TRADES**
    To tackle the fundamental tradeoff between adversarial robustness and natural accuracy, the authors introduce TRADES. This method is built upon an approach that decomposes error into two sections. First, there is natural error, which is the standard classification error on clean data. Second, there is boundary error, which is the probability that an input lies near the decision boundary, making it vulnerable to adversarial attacks.
    

2.  **A Regularized Loss Function**
    Most adversarial training methods try to improve robustness by minimizing the worst-case loss over adversarially modified examples. TRADES proposes the following optimization framework.
    
$$
\min_f E \left[ \phi(f(X)Y) + \max_{X' \in B(X, \epsilon)} \phi\left(\frac{f(X) f(X')}{\lambda} \right) \right]
$$

  
Where X is the natural input sample, Y is the true label, X’ is the adversarial example, ϕ(f(X)Y) is the standard classification loss, the max portion regularizes the model and tries to encourage it to make similar predictions for clean and adversarial examples, and λ is a hyperparameter that determines the trade-off between accuracy and robustness. A higher λ means more robust, but less accurate. A lower λ means more accurate, but less robust. This structured approach ensures that the model maintains high accuracy on clean data, but leans to push decision boundaries away from data points, improving robustness.

  

3.  **How TRADES is Implemented**
    TRADES is implemented using adversarial training, but instead of directly minimizing adversarial loss, it optimizes balance between natural accuracy and smoothness of decision boundaries. First, the authors start with a clean batch of training samples (X,Y). Then, they generated adversarial examples X’ by maximizing the regularization term using Projected Gradient Descent (PGD). Next, they updated the model parameters using a combination of the standard classification loss and the robustness regularization loss. Finally, they repeated until convergence, gradually refining the model to balance accuracy and robustness.
    

  
  
  
  

## Key Findings

-  **Trade-Off Between Robustness and Accuracy**
    

	- The study mathematically proves that adversarial robustness inherently conflicts with clean accuracy. This means that making a model more resistant to adversarial attacks often leads to a decrease in its performance on clean data.
    
	- Experimental results on CIFAR-10 show that as robustness increases, natural accuracy drops. When robustness was increased from 47.04% to 56.61% under PGD attacks, natural accuracy fell from 87.30% to 84.92%.
    
	-  On MINST, however, TRADES successfully maintains 99.48% natural accuracy while achieving 96.07% robust accuracy, demonstrating its effectiveness on simpler datasets.
    

- **Trades Achieves High Robust Accuracy**
    

	-  The TRADES framework outperforms standard adversarial training by striking a better balance between robustness and clean accuracy.
    
	-  On CIFAR-10, a standard-trained model achieves a natural accuracy 95.29% but is completely vulnerable to adversarial attacks, with 0% robust accuracy under PGD attacks.
    
	-  Using Madry et al.’s adversarial training, robust accuracy improves significantly to 47.04%, but natural accuracy drops 87.30%.
    
	-  TRADES further optimizes this trade-off, achieving 56.61% robust accuracy under PGD attacks – a nearly 10% improvement over Madry et al., while maintaining a strong natural accuracy of 88.64%.
    
	- TRADES also won 1st place in the NeurIPS 2018 Adversarial Vision Challenge, surpassing the runner-up by 11.41% in mean perturbation distance, proving its real-world effectiveness in adversarial defense.
    

## Critical Analysis

### Strengths
    

-   **Mathematical Rigor** 
The paper provides a theoretically grounded explanation of the tradeoff between robustness and accuracy. It decomposes robust error into natural classification error and boundary error. Allowing for a formalized understanding.
    
-   **Empirical Validation** 
The paper includes extensive experiments across multiple datasets (MINST, CIFAR-10, and Tiny ImageNet) and adversarial attack types (PGD, FGSM, C&W, etc.).
    
-   **Scalability and Practicality**
Unlike some adversarial defense techniques that are computationally prohibitive, TRADES is efficient enough for large-scale datasets like Tiny ImageNet. It also introduces a regularization parameter (λ), allowing practitioners to control the trade-off between robustness and accuracy, making the method more adaptable to different applications.
    
-   **Potential for Semi-Supervised Learning**
The paper highlights how TRADES can be extended to semi-supervised learning, meaning it has the potential to leverage unlabeled data to improve robustness further, a key advantage of in real-world scenarios where labeled data is limited.
    

### Weaknesses
    

-   **Drop in Natural Accuracy**
While TRADES improves robustness, it still sacrifices clean accuracy. For example, on CIFAR-10, the best TRADES model improves robust accuracy, but natural accuracy drops. This trade-off may limit adoption in applications where high clean accuracy is critical, such as medical AI.
    
-   **Hyperparameter Sensitivity**
The effectiveness of TRADES depends heavily on tuning the regularization parameter. Choosing the wrong value could lead to suboptimal robustness or unnecessary accuracy loss. The paper doesn’t explore how well TRADES generalizes across different datasets when the parameter is varied, making it unclear how easy it is to deploy in practice.
    
-   **Limited Evaluation Beyond Image Classification**
The study focuses exclusively on image datasets, leaving questions about its effectiveness in other domains such as speech recognition, natural language processing, or cybersecurity.
    

### Potential Biases
    

-   **Dataset Selection Bias**
    The paper evaluates TRADES primarily on benchmark datasets such as MINST, CIFAR-10, and Tiny ImageNet, which are well-curated and balanced. However, real-world datasets often contain class imbalances, skewed distributions, and noisy labels. The absence of testing on diverse datasets raises concerns about how TRADES would perform in applications with underrepresented classes.
    

-   **Lack of Fairness Analysis**
    The paper doesn’t investigate whether TRADES has differential impacts across demographic groups. Without analysis of fairness metrics, it is unclear whether TRADES disproportionately harms or benefits certain groups in real-world settings.
    

-   **Focus on lp-Norm Attacks**
    TRADES is evaluated against l-bounded perturbations, which are widely used in adversarial robustness research but doesn’t cover the full spectrum of adversarial threats. Unrestricted attacks, like geometric transformations and data poisoning, aren’t tested, meaning the results may not generalize well to non-lp-bounded adversarial strategies.
    

### Ethical Considerations
    

-   **Security vs. Accessibility Trade-Off**
    While TRADES enhances adversarial robustness, making AI systems harder to fool, it also raises concerns about an arms race between attackers and defenders. As adversarial defenses improve, attackers may develop stronger adaptive attacks, making security research more challenging.
    

-   **Fairness and Bias in Robust Models**
     The paper doesn’t examine whether adversarial robustness disproportionately affects certain demographic groups. Research suggests that robust models can unintentionally amplify biases, meaning that underrepresented groups may experience more classification errors when robustness is enforced.
