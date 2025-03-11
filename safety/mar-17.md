# Safety - Poisoning

## [Certified Defenses for Data Poisoning Attacks](https://arxiv.org/abs/1706.03691). Jacob Steinhardt, Pang Wei Koh, Percy Liang, 2017.

### Introduction and Motivations

The most critical part of security of machine learning algorithms is the training data, which can be targeted via data poisoning. The focus of this work is on poisoning attacks that target classification models, which are hard to defend against due to the massive amount of possible attack variations. The researchers propose an approach to understanding the entire space of attacks a model could be vulnerable to, in this case pertaining to binary SVMs.

### Methods

The researchers administer a white-box attack to insert poisoned data into the training set in order to distort the centroid of each class. 
The goal of the poisoning attack is to increase the model’s test loss by strategically modifying the training data. This is done by injecting εn poisoned samples 𝐷𝑝, which shift the class centroids and distort the decision boundary. Mathematically, the attacker aims to solve:

$
\max\limits_{D_p} \mathbf{L}(\hat{\theta}) =
\max\limits_{\mathcal{D}_p \subseteq \mathcal{F}} \min\limits_{\theta \in \Theta} \frac{1}{n} L(\theta; \mathcal{D}_c \cup \mathcal{D}_p) \overset{\text{def}}{=} \mathbf{M}.
$

where L(θ) is the test loss. By shifting the estimated class means, the attacker forces the model to learn a biased decision rule, leading to higher misclassification rates.

Visualized below are the poisoned datasets along with two defense mechanisms, “sphere” and “slab” defense which remove outliers based on different mathematical criteria. ![image](images/mar17/certified_fig1.png)

The sphere defense removes data points that deviate too far from the class centroid in Euclidean space. The slab defense restricts data points to remain within a certain margin along the decision boundary. These mechanisms ensure that extreme outliers are removed before training.
Both Slab and Sphere defenses can be implemented in two ways: 
1) as a fixed defense using true class means 
2) as a data-dependent defense using empirical means. 

Fixed defenses rely on predetermined thresholds or external knowledge, and data-dependent defenses estimate class means use the available training data. This includes both clean and potentially poisoned samples. This introduces a major vulnerability: if an attacker injects carefully crafted poisoned samples, the estimated class means will shift, leading to a compromised defense mechanism.


#### Bounds on test loss
the researchers’ goal is to maximize the test loss using their attack. They provide mathematical proofs to bound the worst-case test loss given (epsilon n) poisoned elements. 

#### Experiment
Researchers target models trained on the Dogfish and MNIST-1-7 datasets, with the conditions of each experiment being the fraction of poisoned data added (epsilon) and the type of defense administered. One defense mechanism was an oracle defender that had knowledge of the true class means. The other was a data-dependent defense that blindly used the empirical means of the poisoned dataset.

### Key Findings
For the MNIST and Dogfish datasets, the oracle defenders supplied with the true means of each class were highly effective at thwarting the poisoning attack; even after adding epsilon=30% of poisoned data, the test loss remained below 0.1. 
![image](images/mar17/certified_fig2.png)

Researchers added a text classification task on the Enron spam email dataset and the IMDB sentiment corpus. On these datasets, the attack was more effective against oracle defense and led to large increases in test loss as epsilon increased.

The data-dependent defense was much weaker than the oracle defense - as shown in the figure below demonstrating a maximum test loss U(theta) increasing at a higher rate than against any other defense mechanism as epsilon increases.
![image](images/mar17/certified_fig4.png)

### Critical Analysis
#### Strengths
- The paper justifies its design choices well and succinctly defines its metrics for test loss, max test loss, and minimax loss. With these definitions in place, the paper is able to prove and derive additional claims effectively.
- Researchers are able to demonstrate compelling results by narrowing the scope of their experimentation to a restricted number of datasets and defense mechanisms, with significant differences in test results between the control and experimental groups.
- The paper supplies all of its code and data for replicating experiments.

#### Weaknesses
- To establish their poisoning attack’s real-world effectiveness, researchers brush off its reliance on white-box knowledge of the models and their training data as they claim model privacy / obscurity is an ineffective line of defense. The paper could benefit from some additional argumentation on how model details and training data can be hacked by an attacker as the researcher’s entire attack lies on the premise that all of this information is somehow available.
- The paper’s results demonstrate an effective attack on specific scenarios (e.g. MNIST-1-7 with a data-dependent defense). However, this attack is not effectively shown to be generalizable on more common & complex classification tasks such as ImageNet which many models at the time were evaluated with.

#### Ethical considerations
- Releasing this code to the public comes with significant ethical considerations as malicious actors could use this attack methodology to target ML algorithms in the wild. Moreover, no thorough defense for the attack is shown, leaving ML developers on their own to come up with safeguards against this new poisoning attack.