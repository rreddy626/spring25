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
   
These ideas fall under the notion of meta-learning, which means that a “model develops a broad set of skills and pattern recognition \
abilities at training time and then uses those abilities at inference time to rapidly adapt or recognize the desired task.” 
   
