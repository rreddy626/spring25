# Safety - Poisoning

## [Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](https://arxiv.org/abs/1804.00308). Jagielski et al. 2018.

### Introduction and Motivations

Training data poisoning attacks occur when attackers inject a small amount of corrupted data points into the training process for a machine learning (ML) model. These attacks are becoming more common as an increasing number of ML models require online training so that they are updated with new incoming training data. Furthermore, defending against training data poisoning attacks is challenging with current defensive techniques. 

Regression models learn to predict a response variable based on several predictor variables while minimizing the loss. The impact of such poisoning attacks on linear regression models and how to design stronger countermeasures has yet to be explored in depth. This work conducts one of the first studies on poisoning attacks and their defenses on linear regression models. 

### Methods
This paper considers four different linear regression models: Ordinary Least Squares (OLS), ridge regression, LASSO, and elastic-net regression. It evaluates their novel poisoning attacks and defense algorithm on three regression datasets on health care, loans, and housing prices and compares them with the baseline gradient descent attack (BGD). The study evaluates the metrics of success rate of the poisoning attack by comparing the corrupted model and legitimate model Mean Squared Error (MSE) as well as the running time of the attack. 

*Adversarial model*: The adversary’s goal is to modify predictions made by the learning model on new data by corrupting the model during the training process. Attacks can be under white-box or black-box settings. In white-box attacks, the adversary has knowledge of the training data, feature values, learning algorithm, and the trained parameters. On the other hand, in black-box attacks, the adversary has knowledge of the feature values and learning algorithm, but not the training data and trained parameters. An adversary’s capability is upper bounded by the number of poisoning points that can be injected into the training data. Therefore, the adversary is usually assumed to only control a very small portion of the training data. Finally, the poisoning attack strategy can be formalized as a bilevel optimization problem. 

*Optimization-based poisoning attack (OptP)*: Previous poisoning attacks were developed for classification problems, limiting their effectiveness against regression models. Optimization-based poisoning attacks work by iteratively optimizing on a single poisoned data point at a time through gradient ascent. This paper adapts the optimization-based poisoning attack for regression tasks by utilizing two initialization strategies (inverse flipping and boundary flipping) and jointly optimizing both the feature values and their associated response variables. The authors also construct a baseline gradient descent (BGD) attack for regression. 

*Statistical-based poisoning attack (StatP)*: Jagielski et al. also develop an attack that produces poisoned data points with a similar distribution as the training data. This attack requires estimations of the mean and covariance from the training data distribution and is agnostic to the regression algorithm, its parameters, and the training set. As a result, it requires minimal information and is also significantly faster than optimization-based poisoning attacks. However, they are generally slightly less effective. 

*Defenses*: Existing defenses against poisoning attacks can be classified as either noise-resilient or adversarially-resilient. Noise-resilient regression approaches identify and remove any outliers from the dataset. However, an adversary can just generate poisoned data points that are very similar to the training data that can still mislead the model. Meanwhile, adversarially-resilient regression approaches generally have provable robustness guarantees, but the strong assumptions about the data and noise distributions made are usually not satisfied in practice. To improve upon existing defenses, the authors of this work propose the TRIM algorithm which identifies training data points with the lowest residuals relative to the regression model and disregards the points with large residuals. TRIM terminates once the algorithm converges and the loss function reaches a minimum. It is proved that TRIM terminates in a finite number of iterations.

### Key Findings

#### Attack Evaluation

1) Which optimization strategies are most effective for poisoning regression?

OptP outperforms BGD by a factor of 6.83 in the best case, achieving MSEs by a factor of 155.7 higher than the original models. Each dimension of the optimization framework (initialization strategy, optimization variable, objective of optimization) is crucial to generating a successful attack. 

2. How do optimization and statistical attacks compare in effectiveness and performance?

Generally the optimization-based attacks (OptP and BGD) outperform the statistical-based attack (StatP). This is expected because StatP uses less information about the model training when compared to the other attacks. However, StatP is still a reasonable attack to use if an attacker has limited knowledge and runs faster than the optimization-based attacks, highlighting the tradeoff between effectiveness and computational resources. 

3. What is the potential damage of poisoning in real applications?
To analyze the potential damage of poisoning attacks in real applications, the authors examined poisoning on the health care dataset. The new poisoning attacks can cause the linear regression models to significantly change the predicted drug dosage for patients even with a small percentage of poisoned data points. 

4. What are the transferability properties of our attacks?

Optimization-based and statistical-based poisoning attacks both have good transferability properties. There are minimal differences in accuracy when used on different training sets. Some exceptions to these results require further research. 

#### Defense Evaluation
1. Are known methods effective at defending against poisoning attacks?

Existing defenses are not very effective at defending against the novel poisoning attacks introduced in this paper. Furthermore, there is the possibility that they may increase MSEs over unpoisoned models.

2. What is the robustness of the new defense TRIM compared to known methods?

Compared to known defenses, TRIM is much more effective at defending against all poisoning attacks. Unlike previous approaches, TRIM also improves upon the MSEs. 

3. What is the running time of various defense algorithms?

The various defense algorithms all ran within reasonable time, with TRIM running the fastest. 

### Critical Analysis

#### Strengths
- This paper proposes novel poisoning attack and defense methodologies and is the first to contribute to studying model poisoning on linear regression models. 
- The study is detailed and provides a comprehensive evaluation of the poisoning attacks and defenses by evaluating on different kinds of regression models, different datasets, and comparing to a baseline attack and existing methods. 
- The authors not only show that TRIM improves upon existing defenses against poisoning attacks through their experiments, but they also provide provable guarantees which offer theoretical evidence to support why their algorithm works. 
- This work demonstrated how harmful real world applications of poisoning attacks can be in a case study with the health care dataset. Since even a small amount of poisoning in a linear regression model can lead to significantly different and harmful results, future research into defending against poisoning attacks must be done. 

#### Weaknesses
- The contributions of this paper may not generalize to all regression models and datasets because only a select few were analyzed in these experiments. There are several other types of regression models and datasets that were not considered in this paper. 
- It is possible that the proposed poisoning attacks and defense algorithms are not practical to use in the real-world. For example, the optimization-based attack requires more computational overhead. 

#### Potential Biases
In general, there may be potential biases with the dataset, evaluation methods, and overall problem definition. Certain datasets and evaluation metrics may lead to more favorable results for the novel poisoning attacks and defense algorithms while the overall problem definition may make assumptions about the attack scenario that are biased.

#### Ethical Considerations
This paper conducted a case study on a health care dataset to demonstrate real-world implications of poisoning. Datasets like this one may contain sensitive data, resulting in privacy and security concerns when conducting research on attacks. There are also ethical considerations in terms of transparency and dual-use cases. This paper provides the code for their study in a public GitHub repository, but there may be concerns about if the proposed attacks are improved upon by other researchers or attackers with malicious intent. 


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