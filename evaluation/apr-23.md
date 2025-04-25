

## Introduction and Background

Machine learning is now widely used in areas like healthcare, criminal justice, and hiring. As its influence grows, so do concerns about the lack of transparency in how models are built and evaluated. Many models are released without clear information about their purpose, how they were tested, or who might be affected by their failures.

To address this, Mitchell et al.[74] proposed **Model Cards** — a structured way to describe a model’s intended use, performance across demographic groups, and known limitations. This builds on earlier work like **Datasheets for Datasets** by Gebru et al. [75], which focused on dataset transparency. Later, **Data Cards** (Pushkarna et al., 2022 [77]) were introduced as a more practical version for industry use. Meanwhile, research by Birhane et al. [76] found that most ML papers rarely mention ethical concerns. This highlights the importance of tools like Model Cards, which help bring fairness and accountability into everyday ML practice.

## DataSheets vs Data Cards vs Model Cards

| **Aspect**         | **Datasheets**                                                            | **Data Cards**                                                                 | **Model Cards**                                                              |
|--------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Purpose**        | Standardize documentation of datasets to ensure transparency and accountability in dataset creation and use.    | Provide accessible, structured summaries of datasets to support responsible AI, especially in organizational and product development contexts.               | Document trained ML models to inform stakeholders about intended use, performance across groups, and ethical considerations. .       |
| **Focus**          | Captures a dataset’s origin, composition, collection methods, and intended use. It helps in evaluating the appropriateness of a dataset for specific ML tasks.          | Lifecycle-based documentation, including both observable and contextual aspects such as dataset design rationale, known benchmarks, and instructions for use.       | Disaggregated performance evaluations across demographic and phenotypic subgroups.                    |
| **Audience**       | Dataset creators and consumers (especially in academia or research).      | Diverse stakeholders (developers, auditors, policymakers), not necessarily dataset experts. | ML practitioners, developers, policymakers, affected users.                 |


## Key Features of DataSheets, Data Cards and Model Cards
| **Aspect**                  | **Datasheets**                                                                 | **Data Cards**                                                                                 | **Model Cards**                                                                 |
|-----------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Format**                  | Long-form, question-based format.                                              | Modular, structured format using blocks (title, question, input).                              | Short documents (1–2 pages). |
| **Documentation Style**     | Encourages manual, reflective documentation—not automated.                    | Includes fairness-aware evaluations, intentions, usage, and provenance.                         | Includes model details, training data, evaluation methods, use cases, ethical caveats.                              |
| **Purpose/Benefit**         | Useful for reproducibility and mitigating societal bias.                      | Acts as a "boundary object"—shared across diverse roles with interpretive flexibility.          | Promotes fair, transparent deployment of ML models.                         |
| **Lifecycle Use**           | Designed during dataset creation and updated with changes.                    | Adaptable for digital platforms and user-centric design.                                        |       Often complements datasheets by linking to data sources.                                                                           |

## How DataSheets, Data Cards and Model Cards Protect Values Encoded in ML Research

ML research often uplifts certain values—especially performance, generalization, efficiency, and novelty—while neglecting broader social concerns like justice, inclusion, or societal need. This aligns directly with the rationale behind Datasheets, Data Cards, and Model Cards, which were introduced to rebalance priorities in ML systems. 

| **Aspect**             | **Datasheets**                                                                                     | **Data Cards**                                                                                                               | **Model Cards**                                                                                     |
|------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Problem Addressed**  | ML research relies on opaque, pre-established datasets without questioning their origins or biases. | Assumption that data are clean, binary, or inherently truthful.                                                              | ML work rarely discusses ethical ramifications or demographic disparities of models.                |
| **Response/Action**    | Require reflection on data provenance, intended use, consent, collection methods, and limitations.   | Capture both observable and unobservable aspects—labeling choices, context, fairness implications, social impact.            | Document intended use cases, disaggregated performance metrics, ethical concerns, and potential risks. |


## Methodology

This section outlines the specific methodologies used across four foundational papers on transparency in machine learning. While all aim to improve the interpretability, accountability, and ethical deployment of AI systems, each takes a distinct approach grounded in its respective context—ranging from structured design and participatory development to empirical analysis and qualitative annotation.

### Model Cards for Model Reporting [74]

The methodology in this work is based on a design-oriented approach, focused on establishing a standardized format for model documentation. The authors drew on conventions from other fields, such as nutrition labels and medical trial reports, to create a consistent structure for communicating a model’s capabilities, intended uses, limitations, and ethical concerns.

To validate the proposed template, the authors applied it to two example models: a smile detection classifier and a toxicity detection model. These case studies were used to demonstrate how structured documentation could support disaggregated performance analysis—particularly across demographic subgroups—and encourage responsible deployment. The methodology emphasized clarity, modularity, and accessibility, aiming to support stakeholders ranging from model developers to end users and regulators.

### Datasheets for Datasets [75]

This paper employed a human-centered, iterative design methodology to develop a framework for dataset documentation. The authors proposed a set of guiding questions organized around stages of the dataset lifecycle, including motivation, composition, collection, preprocessing, usage, and maintenance.

Initial drafts of the datasheet were tested on real-world datasets and refined through collaboration with industry practitioners and legal experts. The methodology emphasized manual, reflective documentation over automation, encouraging dataset creators to explicitly consider issues such as bias, data provenance, and appropriate use cases. The result was a flexible and extensible template meant to support transparency and accountability across diverse data domains.

### The Values Encoded in Machine Learning Research [76]

This paper used a qualitative content analysis methodology to examine the implicit values present in machine learning research. The authors selected 100 of the most cited NeurIPS and ICML papers across two time periods (2008–2009 and 2018–2019) and developed a custom annotation scheme to code for the presence of various value-related dimensions.

Over 3,500 sentences were manually annotated by multiple reviewers, using a hybrid inductive and deductive coding process. Annotations captured whether papers justified their work based on societal needs, prioritized performance or novelty, discussed potential harms, or acknowledged structural impacts. The analysis revealed a strong emphasis on technical performance and a lack of attention to ethical implications or social responsibility. The methodological rigor was supported by inter-annotator agreement scores and reflexive reporting on the annotation process.

### Data Cards for Responsible AI [77]

This paper applied a participatory, production-focused methodology to develop a scalable dataset documentation tool tailored for use in industry settings. Over the course of two years, the authors collaborated with twelve teams inside a large technology company to co-create 22 “Data Cards” spanning a variety of modalities (e.g., text, image, audio, and tabular data).

The methodology combined human-centered design, stakeholder mapping, and empirical feedback collection. A typology of stakeholders—producers, agents, and reviewers—was used to define documentation needs at different stages of the dataset lifecycle. The authors also conducted a MaxDiff survey (n = 191) to identify and prioritize the most valuable fields to include. A supporting framework, OFTEn, was introduced to help authors capture both observable characteristics and unobservable rationales. The resulting Data Cards were tested and iteratively refined based on real-world usability, with an emphasis on scalability, consistency, and accessibility.

---

Each of these methodological approaches contributes uniquely to the broader effort to operationalize transparency in AI systems. From structured design and reflective documentation to large-scale empirical evaluation and embedded industry practice, they represent complementary paths toward more responsible machine learning.s. Most papers only justify how they achieve their internal, technical goal; 68% make no mention of societal need or impact, and only 4% make a rigorous attempt to present links connecting their research to societal needs. One annotated paper included a discussion of negative impacts and a second mentioned the possibility of negative impacts. 98% of papers contained no reference to potential negative impacts. . Comparing papers written in 2008/2009 to those written in 2018/2019, ties to corporations nearly doubled to 79% of all annotated papers, ties to big
tech more than tripled, to 66%, while ties to universities declined to 81%, putting the presence of corporations nearly on par with universities.

Emphasizing performance is the most common way by which papers attempt to communicate their contributions, by showing a specific, quantitative, improvement over past work, according to some metric on a new or established dataset. For some reviewers, obtaining better performance than any other system—a “state-of-the-art” (SOTA) result—is seen as a noteworthy, or even necessary, contribution. 

The prioritization of performance values is so entrenched in the field that generic success terms, suchas "success", "progress", or "improvement" are used as synonyms for performance and accuracy. However, one might alternatively invoke generic success to mean increasingly safe, consensual, or participatory ML that reckons with impacted communities and the environment. In fact, "performance" itself is a general success term that could have
been associated with properties other than accuracy and SOTA.

## Key Findings 

### The Values Encoded in Machine Learning Research

The top values that are mentioned in research papers are : performance (96% of papers), generalization (89%), building on past work (88%), quantitative evidence (85%), efficiency (84%), and novelty (77%). Values related to user rights and stated in ethical principles appeared very rarely if at all: none of the papers mentioned autonomy, justice, or respect for persons.

![](images/apr23/Paper_containing_value.png "Percent of papers containing value")

Comparing papers written in 2008/2009 to those written in 2018/2019, ties to corporations nearly doubled to 79% of all annotated papers, ties to big
tech more than tripled, to 66%, while ties to universities declined to 81%, putting the presence of corporations nearly on par with universities. In the next section, we present extensive qualitative examples and analysis of our findings.
![](images/apr23/corporate_funding.png "corporate funding")




Analysis shows substantive and increasing corporate presence in the most highly-cited papers. In 2008/09, 24% of the top cited papers had corporate affiliated authors, and in 2018/19 this statistic more thandoubled to 55%. Furthermore, we also find a much greater concentration of a few large tech firms, such as Google and Microsoft, with the presence of these "big tech" firms (as identified in [4]) increasing nearly fourfold, from 13% to 47%.
![](images/apr23/corporate%20and%20big%20tech%20author%20affliliations.png "big tech funding")



Emphasizing performance is the most common way by which papers attempt to communicate their contributions, by showing a specific, quantitative, improvement over past work, according to some metric on a new or established dataset. For some reviewers, obtaining better performance than any other system—a “state-of-the-art” (SOTA) result—is seen as a noteworthy, or even necessary, contribution. 

The prioritization of performance values is so entrenched in the field that generic success terms, such as "success", "progress", or "improvement" are used as synonyms for performance and accuracy. However, one might alternatively invoke generic success to mean increasingly safe, consensual, or participatory ML that reckons with impacted communities and the environment. In fact, "performance" itself is a general success term that could have
been associated with properties other than accuracy and SOTA.


## Critical Analysis

### Model Cards for Model Reporting[74]:
Model Cards present a clear, structured framework for documenting machine learning models, with a strong emphasis on transparency, fairness, and accountability. By encouraging performance evaluation across different demographic, cultural, and environmental groups—including intersectional categories such as race and gender—Model Cards help surface potential biases before deployment. This enables stakeholders to assess model behavior in socially sensitive contexts and promotes more responsible AI practices. The framework serves as a foundation for standardizing ethical model reporting, making it easier for developers, policymakers, and users to understand a model’s limitations and intended use.

Despite its valuable contributions, the Model Cards framework mainly targets supervised learning models and provides little guidance for unsupervised or generative systems. Its practical implementation can also be challenging due to limited discussion on tooling or integration into production workflows. Moreover, several recommendations—such as detailed intersectional performance breakdowns—assume access to accurate ground truth labels for sensitive attributes, which may not always be available, reliable, or ethical to collect. These challenges can hinder widespread adoption, especially in real-world settings where such demographic data is scarce or legally restricted.

### Datasheets for Datasets [75]:
This framework enhances transparency, accountability, and informed decision-making by dataset consumers, and also encourages dataset creators to critically reflect on the data collection and usage processes. It directly addresses the risks of societal biases and unintentional misuse of datasets in high-stakes domains by fostering a culture of careful documentation. The authors acknowledge variability across domains and institutions, promoting a flexible rather than rigid approach to datasheet content.

The authors acknowledge that the questions in the datasheet are not exhaustive or universally applicable, which could lead to inconsistent implementation across different domains or organizations. Additionally, the process is not designed to be automated, which may hinder widespread adoption due to the significant manual effort required. Although the authors worked with legal teams and revised questions to avoid direct regulatory references, compliance and liability concerns may still arise, especially for organizations in heavily regulated industries.

### The Values Encoded in Machine Learning Research[76]:
The Strengths of paper include The paper introduces a fine-grained annotation scheme to analyze the values present in ML research papers, which is both novel and methodologically rigorous. It offers a unique lens for studying how scientific research embeds and amplifies particular value systems, a contribution especially important as ML increasingly affects real-world systems and lives. Additionally, By analyzing 100 highly cited ICML and NeurIPS papers, the authors provide compelling evidence that performance, efficiency, and generalization dominate as values—often at the expense of ethical considerations or societal impact. This is backed by strong data and textual analysis, offering a grounded critique of how ML may reinforce existing power hierarchies.

Talking about the weaknesses of the paper, while the critique of performance-centric evaluation is valid, the paper sometimes risks trivializing the legitimate role of values like performance and generalization in driving scientific progress. There’s a fine line between exposing social blind spots and underappreciating the technical sophistication and relevance of these priorities. Also, The paper seems to adopt a subtly critical tone toward corporate participation and scale-driven research, potentially framing "Big Tech" involvement as inherently suspect. While valid concerns are raised (e.g., centralization of power), the framing may paint corporate contributions with too broad a brush, overlooking nuanced or beneficial roles these actors might play in pushing ML boundaries responsibly.

### Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI [77]:
The framework is designed to be detailed, modular, and extensible, making it adaptable across various dataset modalities (e.g., text, image, audio) and applicable in both research and industry contexts. A major contribution is its emphasis on intelligibility and accessibility, enabling non-technical stakeholders to effectively interpret dataset details. By promoting shared mental models and surfacing “known unknowns,” Data Cards help reduce knowledge asymmetries among stakeholders and support better decision-making practices in AI development.

Much of the documentation relies on manual input, which can lead to inconsistencies, increased workload, and fragmented templates due to the forking of existing Data Cards. Additionally, the framework's effectiveness depends heavily on the producer's familiarity and diligence, making the quality of documentation variable and subjective. Furthermore, the paper touches on a common dilemma—disclosing enough to support transparency while avoiding risks like exposing systems to adversarial attacks; striking this balance is nuanced and context-dependent.

## References

[74] [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). Mitchell et al., 2018.  
[75] [Datasheets for Datasets](https://arxiv.org/abs/1803.09010). Gebru et al., 2018.  
[76] [The Values Encoded in Machine Learning Research](https://arxiv.org/abs/2106.15590). Birhane et al., 2021.  
[77] [Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI](https://dl.acm.org/doi/10.1145/3531146.3533231). Pushkarna et al., 2022.