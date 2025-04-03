# [Differentially Private Empirical Risk Minimization](https://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf). Chaudhuri et al 2011. 


## Introduction 

Anonymization and data aggregation are not enough for data privacy, as datasets can be cross-referenced or reverse-engineered to reveal sensitive personal information on the training data. Instead, researchers have developed ε-differential privacy methods, which mathematically guarantees the following definition of privacy: changing a single data point does not shift the output distribution “by too much” (some given epsilon degree). This makes it difficult to infer the value of any particular data point that a model was trained on, ensuring user data is kept safe. This particular paper dives into the empirical risk minimization (ERM) framework, introducing two methods of differential privacy on classification tasks. ERM’s goal is to minimize the average over the training data of the prediction loss (with respect to the label) of the classifier in predicting each training data point. The two methods, output perturbation and objective perturbation, follow this framework by introducing noise in two different parts of the ERM pipeline—the output of the standard ERM algorithm and regularized ERM objective function prior to minimizing loss, respectively. The researchers also follow end-to-end privacy, ensuring that each step in the model learning process remains private, since intermediate steps such as training and parameter tuning can cause additional risks of privacy violations. 

## Methods 

## Key Findings 

## Critical Analysis 

### Strengths 

Objective perturbation is the most important contribution. Instead of adding noise after training, it adds noise directly to the objective function before training starts. This means the method needs less noise overall. The result is a model that still performs well but keeps data private. This helps balance privacy and accuracy, which is often a hard problem in privacy-preserving machine learning. The paper proves that both methods give ε-differential privacy guarantees. It also gives bounds on generalization error, showing that the model trained with noise will still do well on unseen data. These theoretical results are solid and make the work trustworthy.

The authors show that their methods can be used with logistic regression, support vector machines, and kernel methods. These are common models, so the paper is useful to many real-world applications. They also talk about how to protect privacy during hyperparameter tuning, which is an important but often overlooked part of the learning process. 

### Weaknesses 

The methods only work when the loss function and the regularizer are both smooth and strongly convex. This means the approach cannot be used with L1 regularization or with hinge loss. These are common in many real models. So, the method may not work well in practice for all tasks. The way the paper applies privacy to kernel methods is not very efficient. It may lose some performance or require extra steps like using random feature mappings. This could make the model slower or harder to train. The experiments are limited. The datasets are not very large or complex. The paper does not test on modern deep learning models. So, it is hard to say how well the method works in more advanced settings.

### Potential biases 

If the training data contains imbalance, the model might learn patterns that do not generalize well to those groups. Differential privacy adds noise, which may hurt performance more on small subgroups. This can lead to unequal accuracy across different users.

The model is optimized to reduce average loss. But average accuracy does not always reflect fairness. The noise added for privacy may hide these fairness issues. This is important in applications like healthcare or education, where different groups may be affected differently.

### Ethical considerations 

# [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133). Abadi et al, 2016 


## Introduction

- Model training requires large datasets which are often crowdsourced and can contain sensitive information.  
- It is critical to balance machine learning training objectives with privacy-preserving goals.  
- Researchers improve the efficiency of differentially private training in Tensorflow, demonstrating that by tracking detailed information of privacy loss, the overall privacy loss can be estimated accurately.

## Methods

1. Differentially private SGD:

   ![][image1]

   1. Controls the influence of the training data on the weights of a neural network by clipping gradient norms at different thresholds across layers of the network  
   2. In typical differential privacy fashion, they add noise to the data to improve anonymity. Specifically, they add gaussian noise to the gradients before back propagation.  
2. Moments accountant

	Researchers introduce a “moments accountant” which is used to control training and continuously estimate the privacy loss. They propose this as an improvement (a tighter upper bound) to an expected privacy loss calculated by the “strong composition theorem”, which they provide as: 

![][image2]

1. The moments accountant instead uses the following privacy loss (c)

   ![][image3]

   1. o: outcome, M: mechanism, aux: auxiliary input, (d, d’): neighboring datasets  
   2. The accountant continuously updates the state of training with this privacy loss estimate.  
   3. Researchers use a privacy budget to balance training with two variables  
      1. Epsilon: measure of privacy loss. Higher epsilon means stronger privacy but weaker statistical accuracy  
      2. Delta: probability of a privacy breach, secondary parameter that works with epsilon to balance privacy and utility  
3. Hyperparameter tuning  
   1. Researchers hypothesize that using their differential privacy mechanism, the range of values for hyperparameters would decrease and less tuning would be necessary. They propose that the learning rate, for example, would not need to decay like it would in traditional ML training. This is probably because the noise added by the privacy mechanism would hinder convergence at a lower learning rate, when the model would be trying to make smaller adjustments to the training data.

The researchers evaluate their DP-SGD algorithm on the MNIST and CIFAR-10 image classification tasks. They also add a layer of PCA projection to improve dimensionality reduction.

## Key Findings

Researchers achieve 97% training accuracy on MNIST and 73% training accuracy on CIFAR-10 with epsilon=8, delta=10^-5 differential privacy.  
When compared to the expected privacy loss returned by their strong composition theorem, researchers find the moments accountant provides a tighter bound on epsilon.![][image4]  
Generally, training accuracy increases as epsilon and delta constraints are relaxed, with higher values of delta steepening the accuracy vs. epsilon tradeoff as observed in Figure 4 below.  
![][image5]

## Critical Analysis

### Strengths

Researchers reach a good balance between privacy and accuracy on basic image classification tasks. They also provide a robust Tensorflow model design that can address both objectives. The models’ ability to converge with additional non-convex, privacy-focused objectives is also impressive.

### Weaknesses

The results section could include more experiments comparing with non-privacy preserving approaches to illustrate the performance difference between the differentially private model and baseline model. Also, the task itself is rudimentary, and more sophisticated benchmarks could illustrate a more impactful usage of the DP-SGD algorithm, such as those in the natural language processing or generative domain. The researchers should also report more robust metrics such as F1 to improve the validity of their results.

### Potential biases

This research was conducted by Google, so it is influenced by Google’s interests in the AI field. For example, researchers implemented all experiments using Google’s Tensorflow, whereas PyTorch is the more widely used deep learning framework in ML research.

### Ethical considerations

# [Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data](https://arxiv.org/abs/1610.05755). Papernot et al, 2016 

## Introduction

## Methods

## Key Findings

## Critical Analysis

### Strengths

### Weaknesses

### Potential biases

### Ethical considerations
