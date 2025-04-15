# LLMs: Privacy in LLMs

## [Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory](https://arxiv.org/pdf/2310.17884)

### Introduction and Motivations:

Large Language Models (LLMs) have proven to be extremely useful in understanding and generating text for certain tasks, and this is achieved with the help of finding rich, high-quality text corpora that can be used to train and test these models. Recently, research has been conducted to quantify the extent to which LLMs can protect against the leakage of private information. While the center of this research has largely focused on the training data that is used to train the models, little research has been conducted in order to assess whether LLMs have the capacity to decipher between public and private information at inference time (i.e. A lot of pre-existing work has focused on the training data itself). The purpose of this paper is to determine the answer to one simple question: "Can LLMs demonstrate an equivalent discernment and reasoning capability when considering privacy in context?" The authors of the paper achieve this by designing a multiple-tiered benchmark called ConfAIde.

Data memorization can provide key insights as to how a model "learns" its information, and metrics have been developed in order to quantify this. But little has been done to focus on a concept referred to as "contextual privacy." This work is largely based off the Contextual Integrity theory, which was introduced by Helen Nissenbaum. This defines the flow of information, and notes what is considered an "appropriate" flow of information between individuals/groups. As provided as an example in the text, if a healthcare provider provides your medical history to an insurance company for marketing purposes, this would be considered a violation of contextual integrity. This work seeks to examine how well LLMs can do in determining whether private information can be shared or not, by attempting varying complex tasks in multiple-tiers.

As previously mentioned, previous work has focused primarily on assessing whether training data can be leaked. This has primarily been researched through the lense of Differential Privacy by hiding aspects of the training data, but this work looks at this from an inference-time perspective. To determine what is considered "acceptable" from an information flow perspective, the paper makes use of a social reasoning concept referred to as "Theory of Mind." Theory of Mind is defined as "the ability to comprehend and track the mental states and knowledge of others." This plays an integral role in setting the context when it comes to making privacy-related decisions.

### Methods:

ConfAIde consists of four different tiers.

### Key Findings:

### Critical Analysis: