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

### Key Findings:

**Language Modelling, Cloze and Completion Tasks:**

(enter figure here)

The following table outlines GPT-3 results on Language Modelling, Cloze, and Completion tasks. The LAMBADA dataset assesses the modelling of long-range dependencies in the form of cloze-type tasks (i.e. Fill in the blank tasks). This is to ensure that with an increase in the number of model parameters within an LLM, long-range dependency accuracies can remain intact. GPT-3 appears to beat the State-Of-The-Art (SOTA) standard for LAMBADA test accuracy across all methods. GPT-3 achieves 76.2%, 72.5%, and 86.4% accuracy across Zero-Shot, One-Shot, and Few-Shot methodologies respectively. Similarly, the HellaSwag dataset involves picking the best “ending” to a story/set of instructions, and the StoryCloze dataset selects the correct ending for five-sentence stories. GPT-3 does not beat the SOTA accuracy on either of these datasets but performs reasonably close to the SOTA accuracy outlined in the table above. 

**Closed Book Question-Answering:**

The following outlines GPT-3's ability to answer Closed Book Question Answering. To assess this concept, GPT-3 is evaluated against three different datasets: Natural Questions, WebQuestions, and TriviaQA. The results are outlined below in comparison to other Machine Learning models: 

(enter figure here)

**Translation Tasks:**

Language Translation remains an integral problem that has proven to be tough to solve with Machine Learning. Previously, Neural Machine Translation (NMT) methods have typically been known as the gold standard for Machine Translation models. It becomes important to note that while training the GPT-2 LLM, a filter was applied to ensure that all training data utilized for the model was exclusively in English due to parameter constraints. Now that the number of parameters has scaled up, the authors of the paper wished to test whether GPT-3 could do a better job with Machine Translation tasks. Therefore, GPT-3's training data (i.e. A derivative of the Common Crawl dataset) includes about 7% text that are in different languages. The following table notes results across SOTA supervised and unsupervised NMT models, and compares them to GPT-3 based approaches: 

(enter figure here)

The following scores provided are Bilingual Evaluation Understudy (BLEU) scores, which are typically used to assess the quality of language translation. Generally, GPT-3 combined with the different methodologies do not beat the SOTA models. However, few-shot GPT-3 approaches perform relatively well when translating into English, as opposed to translating from English. Furthermore, there appears to be a performance skew based on the language of interest. GPT’s English to Romanian translation is relatively weak in comparison to other languages and can potentially be attributed to using a tokenizer specifically designed for the English language.  

**Winograd-Style Tasks:** 

This task is quite unique from the others that have been discussed. Winograd-Style tasks aim to identify a word that a pronoun refers to, particularly when the pronoun is “grammatically ambiguous” but clear in common human-to-human interactions. Although fine-tuned models and GPT-3 have done quite well against the Winograd dataset (i.e. SOTA dataset for ambiguous pronoun identification), GPT-3 struggles against other “adversarial” datasets such as Winogrande for pronoun identification. The table below outlines performances across GPT-3 and SOTA models: 

(enter figure here)

**Common Sense Reasoning: **

The paper then seeks to understand how GPT-3 models compare against physical and scentific reasoning tasks. These are relatively unique tasks that ask for much more than understanding NLP as a subfield; there is a sense of reasoning involved as well. The paper tests GPT-3 against PhysicalQA (PIQA), a dataset that contains questions about the physical world, ARC, a dataset that contains exam questions from elementary school through high school science topics, and OpenBookQA, a subset of Question-Answering tasks. Below is a table that outlines GPT-3 results against SOTA fine-tuned models:

(enter figure here)

Although GPT-3 beats SOTA fine-tuned models for the PIQA dataset, it does not produce reasonably close results across the ARC and OpenBookQA datasets. This could indicate potential ”painpoints” in the model itself, where analogous data may have not been considered for training.

**Reading Comprehension:** 

For reading comprehension, the authors decide to make use of five main datasets: CoQA, DROP, QuAC, SQuADv2, RACE-h, and RACE-m. The paper describes CoQA as a free-form conversation dataset, DROP as a dataset designed to “test discrete reasoning and numeracy in the context of reading comprehension,” QuAC as a dataset that is primarily based on student-teacher interactions, SQuADv2 as a dataset primarily designed for generalized Question-Answering scenarios, and RACE as an English-based multiple choice dataset. Below is a table that outlines the F1 score results across all datasets. However, please note that the RACE results report accuracy, not F1 score. 

(enter figure here)

**SuperGLUE Benchmark:** 

The SuperGLUE benchmark was used primarily to aggregate results from popular models such as BERT and RoBERTa using a systematic approach. Below is a table of results that details GPT-3's results against the SuperGLUE benchmark. It is important to note that in the few-shot approach, 32 examples were used:

(enter figure here)

These results highlight some painpoints with the GPT-3 model. In the WiC dataset, the few-shot approach accuracy is significantly less than the fine-tuned solutions. However, in most cases it does appear that the GPT-3 model does a fine job with few-shot approaches across multiple datasets. Furthermore, one key takeaway that the paper highlights is that the GPT-3 model might perform below expectations when comparing two sentences or snippets together. 

**Natural Language Inference:** 

Natural Language Inference (NLI) is primarily concerned with whether a model can decipher the relationship between two sentences in context. This is also a relatively unique approach to metrics, especially since this can be a multi-class classification problem when calculating accuracy in a dataset. To test NLI, the authors of the paper make use of RTE, an NLI dataset that is found in the SuperGLUE benchmark suite, and the Adversarial Natural Language Inference (ANLI) dataset. The graph below outlines results from the ANLI dataset: 

(enter figure here)

**Synthetic and Qualitative Tasks: **

**Key Takeaways:**

### Critical Analysis:

One of the key points that has been reiterated throughout the paper is the concern for data contamination across the training and evaluation datasets. Data that is present both in the training and evaluation datasets gives the model an unfair advantage of improving performance by “memorizing” answers. Therefore, it is integral that while analyzing the training and evaluation datasets that the data does not overlap across both sections. Although the paper highlights that the best attempt was made to de-duplicate across the training and evaluation datasets, the paper acknowledges that there was a code bug that prevented them from finding all existences of overlaps across the two datasets. The resources were not enough to justify the retraining the model, and therefore, the overlap was left as is.  

However, this is likely not a major problem in the grand scheme of things. Although there have been multiple instances where models have performed better because data overlapped between the training and test data, the relative amount of data that was contaminated was insignificant enough to not produce significant differences in reported results. As a note though, it serves to highlight the importance of obtaining and filtering for clean training and evaluation data. 

The paper did a great job by performing a benchmark contamination analysis. To eliminate potentially leaked examples between the training and evaluation datasets, the paper eliminates any instance of a 13-gram overlap between the training and evaluation datasets. I thought this was an extremely strong idea to justify the “cleanliness” of the datasets. The benchark contamination analysis indicated that most datasets that were contaminated did not make a considerable difference when it came to changes in metrics (i.e. Accuracy, F1, BLEU, etc.). However, there were a couple of datasets (i.e. QuAC, DROP, Reversed Words), where there appeared to be a significant change in performance when cleaned. 

One of the key strengths of this paper was how objectively and holistically the results were communicated. In many papers, it is common to see that results are either embellished or misrepresented. However, the paper did an excellent job of acknowledging that GPT-3 has shortcomings in numerous NLP tasks, including text synthesis. GPT-3 is also limited by algorithmic and architectural constraints. This might explain why GPT-3 performs well on certain datasets but fails considerably on some as well (i.e. ANLI, QuAC, RACE, etc.). Other limitations include lack of interpretability, lack of true understanding behind the thought process of few-shot learning, and poor sample efficiency. At the end of the paper, the authors also acknowledge the possibility of misuse, and associations between gender, race and bias. 
