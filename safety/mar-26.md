# Safety: LLMs and Prompt Injection:

## 40: Universal Adversarial Triggers for Attacking and Analyzing NLP:

### Introduction and Motivations:

## 41: Language Models are Few-Shot Learners

### Introduction and Motivations:

Large Language Models (LLMs) have scaled in size, particularly with the introduction of different types of architectures. \
In addition, the idea of training Natural Language Procesing (NLP) models for certain use cases has also evolved over time. \
Instead of fine-tuning model architectures on a specific dataset, what if there was a way for an LLM to decipher between \
different types of tasks without the need to fine-tune the model on specific datasets? This introduces a variety of benefits:

1. Resources
   - The number of resources needed to collect a ground-truth dataset and train/fine-tune a model in order to achieve a certain NLP-related task are significant. By introducing a task-agnostic model, the need to fine-tune a model based on a certain dataset disappears, and scales with the number of parameters utilized in an LLM. We will further explain this phenomenon below.
2. Exploiting Spurious Correlations
   - (TODO)
3. Human-Like Thinking
   - In most scenarios, humans do not require as many examples for many tasks that we ask NLP-based systems to perform. In fact, in most cases, only one or two examples are needed for a human to understand how sentence-completion or reading comprehension tasks work. The unofficial gold standard that many researchers hope to achieve is that one day, NLP systems are designed to act and “think” like humans do.
   
These ideas fall under the notion of meta-learning, which means that a “model develops a broad set of skills and pattern recognition abilities at training time and then uses those abilities at inference time to rapidly adapt or recognize the desired task.” Furthermore, the number of parameters that Machine Learning models have been trained on has also increased significantly over time. As a result, improving the number of parameters that an LLM can be trained on has also brought significant improvements to common NLP-based tasks such as text synthesis. One of the main hypotheses of the paper assesses whether in-context learning will show equivalent improvements with parameter scaling.  

In-Context Learning can be defined as a series of methods that are used to determine how rapidly a model can adapt to specific scenarios that would likely not been seen in the training data in the first place. In other words, it’s a method that can be used to feed context-specific examples to a prompt. To test this hypothesis, the paper assesses the effectiveness of few-shot learning, one-shot learning, and zero-shot learning prompts across different types of models. 

Although there are instances where it is better to fine-tune the model for a specific task, in-context learning strategies show relatively competitive results with certain datasets as well. However, it is not perfect and struggles with Natural Language Inference (NLI) and Reading Comprehension tasks. Furthermore, results found through this paper could potentially be misleading due to data contamination, instances where content found in the training data is also found in the test data as well; this is especially common in large-scale datasets such as Common Crawl. 

### Methods:
Before explaining how the paper tests the effectiveness of in-context learning methods on different LLMs, it is critical to highlight the different approaches that the authors took when evaluating the hypothesis. Below we define the scenarios that the authors tested to evaluate in-context learning: 

1. Few-Shot Approach: 

This is a prompting scenario where a model is given a few demonstrations of the task at the time of inference, but no model weights are to be updated prior to inference. In other words, unlike fine-tuning that requires the alteration of weights, few-shot approaches are primarily used as a “conditioning” approach. Multiple context examples and answers can be given, followed by one final context example. After this point, the model deciphers the answer using the information provided. To evaluate this approach, a random number of examples (denoted by k), are randomly chosen and delimited by 1-2 newline characters. The value of k can increase as long as the number of examples and their respective tokens do not exceed the model’s context window. 

2. One-Shot Approach: 

This is a variation of the few-shot scenario, but instead only one example context and answer are provided for the model. Recall from the introduction that in most scenarios, humans do not require as many examples as a mode

3. Zero-Shot Approach:

In this scenario, we do not provide the model with any demonstrations (i.e. No examples, no example answers). Although this scenario provides convenience and robustness, it might be considered “unfairly hard” by certain standards because the model does not have enough key information to make a proper decision. However,as a counterpoint, in most realistic settings, a zero-shot scenario is extremely close to how humans perform key tasks.

Below is a figure of all the models that were used for testing: 

(enter figure here)

The purpose of testing various LLMs is to assess how the number of parameters affects different learning scenario accuracies, and how well the number of parameters scale over time. It is important to note that GPT-3 and its variants use the same model and architecture as GPT-2. This includes the “modified initialization, pre-normalization, and reversible tokenization.” Furthermore, the Common Crawl dataset is a large-scale dataset that is often utilized when training LLMs because of its plethora of data. However, there are dataset variants whose quality is not up to the caliber of the original dataset. To address this, the paper takes three steps: 

1. Filter the Common Crawl dataset variant with respect to high-quality reference corpora.
2. Perform fuzzy deduplication across documents to reduce redundancy and preserve general data integrity.
3. Add high-quality reference corpora to the training data mix to augment the dataset.

See below for the training data distribution after refinement: 

(enter figure here)
