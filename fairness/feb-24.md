# LLMs: Fairness 

## Introduction and Motivations

### Motivation

Language models increasingly rely on massive web dumps for diverse data. However these resources can be crude in terms of various aspects and can have undesirable content.
As such, resources like Wikipedia , books and newswire are considered as a standard benchmark for good quality and the process typically is referred to as quality filtering. The authors of the paper[17] argue that this quality filter is biased in its approach of choosing a standard or a “good quality” content for a Large Language model to train on, thus the model will be biased as a whole. They support their argument with examples citing that most of the wikipedia authors come from an urban setting, most book authors come from an urban setting with wealthier neighbourhoods and thus considering their language collectively as a good quality benchmark will not be correct. Although these datasets include text from many sources, extensive research suggests that the voices they represent are drawn from a relatively small, biased sample of the population, over-representing authors from hegemonic social positions.They demonstrate this by replicating the quality filter used in GPT-3 and applying it on US high school newspaper articles written by students across the country.
We will now first see some terminologies about fairness and social bias to understand this issue better.

### Definition of Social Bias and Fairness for LLMs [19]

To establish a consistent framework for the often varying definitions of bias in LLMs, this survey [19] synthesizes insights from diverse research sources to present a unified set of definitions.

**Social Group:** A social group $G\in G$ is a subset of the population that shares an identity trait, which may be fixed, contextual, or socially constructed. Several examples include groups based on sex, race, or age.

**Protected Attribute:** A protected attribute is the shared identity trait that determines the group identity of a social group.

**Group Fairness:** Consider a model $M$ and an outcome $ŷ = M(X;\theta)$. Given a set of social groups $G$, group fairness requires (approximate) parity across all groups $G\in G$, up to $\varepsilon$, of a statistical outcome measure $M_{Y}(G)$ conditioned on group membership:

$|M_{Y}(G) - M_{Y}(G')| \leq \epsilon$

The choice of M specifies a fairness constraint, which is subjective and contextual; note that $M$ may be accuracy, true positive rate, false positive rate, and so on.

**Individual Fairness:** Consider two individuals $x, x’ \in V$ and a distance metric $d : V\times V → R$. Let $O$ be the set of outcomes, and let $M : V → \Delta(O)$ be a transformation from an individual to a distribution over outcomes. Individual fairness requires that individuals similar with respect to some task should be treated similarly, such that

$\forall x, x\in V.$ $D(M(x), M(x')) \le d(x, x')$

where $D$ is some measure of similarity between distributions, such as statistical distance.

**Social Bias:** disparate treatment or outcomes between social groups that arise from historical and structural power asymmetries. In the context of NLP, this entails representational harms (misrepresentation, stereotyping, disparate system performance, derogatory language, and exclusionary norms) and allocational harms (direct discrimination and indirect discrimination).

Language has the power to reinforce stero-typesand project social biases onto others. At the core of the challenge is that it is rarely what is stated explicitly, but rather the implied meanings, that frame people’s judgments
about others. In case of offensive language this can be especially be difficult when these implied meanings donot have the words to filter out and thus get passed through. Additionally Most semantic formalisms, to date, do not capture such pragmatic implications in which people express social biases and power differentials in language.The Authors in[18] introduce SOCAIL BIAS FRAMES to tackle this issue.
Let us also discuss some terminologies to discuss this issue and understand it better.

### Bias in NLP Tasks [19]

Bias in NLP tasks can reinforce social hierarchies by embedding stereotypical associations and marginalizing non-standard language variations, leading to representational harms. These biases can manifest subtly, influencing how social groups are categorized and described, which in turn impacts fairness and inclusivity in language technologies. Given the varied nature of NLP applications, bias may emerge in different ways across tasks, shaping how models generate, translate, retrieve, or classify language:

**Text Generation:** In generated task, bias may appear locally or globally. Local bias refers to word-context associations, influencing how specific words are predicted or related within a given context. In contrast, global bias pertains to broader patterns across a span of text, affecting the overall sentiment or thematic direction of the generated content.

**Machine Translation:** Machine translators may default to “selective” words pointing to a particular group in the case of ambiguity.

**Information Retrieval:** Retrieved documents may exhibit similar exclusionary norms.

**Question-Answering:** May rely on stereotypes to answer questions in ambiguous contexts.

**Natural Language Inference:** May rely on misrepresentations or stereotypes to make invalid inferences.

**Classification:** Toxicity detection models misclassify certain dialects more frequently as negative compared to those written in standard language forms.

We now will try understanding these issues and the solutions proposed to mitigate these issue.

# Analysing Bias in Language Ideologies in Text Data Selection [17]

## Methods

The study employs an empirical approach to analyze the behavior of a quality filter used in the training of GPT-3. The researchers replicate this filter and apply it to a newly curated dataset of U.S. high school newspaper articles, enabling them to investigate the demographic and linguistic biases that emerge from such automated filtering.

### Data Collection

#### Primary Dataset: U.S. High School Newspapers

The authors collect a dataset of **1.95 million articles** from **2,483$** U.S. high schools that use a common WordPress-based website template. The dataset is further refined using the following criteria:

- Articles must be in English.
- They should have been published between **2010–2019**.
- The dataset excludes **photo, video, or multimedia content.**
- Only schools with at least **100 articles** are considered to ensure data robustness.

After filtering, the final dataset consists of **910,000 articles** from **1,410 schools** across **1,329 ZIP codes and 552 counties in the U.S.**

#### Demographic Augmentation

To explore demographic biases, the researchers augment the dataset with **geographical and socioeconomic metadata** from sources such as:

- **U.S. Census (2020)** for median home values and educational attainment levels.
- **National Center for Education Statistics (NCES)** for school size and type (public, private, charter, or magnet).
- **MIT Election Lab (2016)** for county-level political leanings (GOP vote share).

### Replicating the GPT-3 Quality Filter[17]

The study replicates the **GPT-3 quality filter**. This filter is a **binary logistic regression classifier** trained to distinguish between **"high-quality"** reference corpora and **Common Crawl** web data.

#### Training Data for the Filter

The classifier is trained using the following datasets:

- **Positive Class ("High Quality" Texts):**
- **80M tokens each from Wikipedia, Books3, and OpenWebText.**
- **Negative Class ("Low Quality" Texts):**
- **240M tokens sampled from the September 2019 Common Crawl snapshot.**

#### Model Training and Validation

The researchers use **Scikit-Learn** to implement the classifier, performing a **100-trial hyperparameter search** with a hashing vectorizer and basic whitespace tokenization. The final classifier achieves:

- **90.4% F1-score**
- **91.7% accuracy** on a **60M-token test set**.

### Analysis of Filtering Behavior

#### Document-Level Biases

The quality filter is applied to the **high school newspaper dataset**, generating a probability score **P(high quality)** for each document. Key findings include:

- **School newspaper articles are more likely to be classified as "low quality"** compared to professional news articles.
- The classifier **favors formal writing styles and longer articles**.
- Topics such as **politics and sports** receive **higher quality scores**, while articles about **campus events or personal experiences** score lower.

#### Demographic Biases

Regression analysis reveals systematic **socioeconomic biases** in quality scores:

- Articles from **wealthier ZIP codes, urban areas, and larger schools** receive **higher quality scores**.
- Articles from **rural schools and schools with lower parental education levels** score significantly lower.
- Schools in **Democratic-leaning counties** tend to have h**igher-scoring** articles than those in Republican-leaning counties.

#### Comparisons with Human Evaluations

To assess whether the quality filter aligns with **human notions of text quality**, the researchers compare its predictions against:

- **Factuality ratings of news sources** (Baly et al., 2018): No correlation found.
- **TOEFL exam essay scores**: The filter is weakly correlated with human scores, but heavily influenced by **the prompt topic**.
- **Pulitzer Prize-winning books**: The filter **prefers non-fiction over poetry and drama**, indicating a bias toward **formal, expository writing styles**.

# Reasoning about Social Implications of Language [18]

## Methods

This research employs an empirical and computational approach to study social bias implications in language. The authors introduce Social Bias Frames (SBF) as a formalism for capturing implicit stereotypes and power dynamics, supported by the Social Bias Inference Corpus (SBIC)—a large-scale dataset annotated with social bias implications.

### Social Bias Inference Corpus (SBIC)

The dataset is constructed by collecting **44,671 posts** from multiple online platforms that frequently contain biased content, including:

- **Reddit** (r/darkJokes, r/meanJokes, r/offensiveJokes, microaggressions dataset)
- **Twitter** (Founta et al., Davidson et al., Waseem and Hovy datasets)
- **Hate sites** (Gab, Stormfront, Banned Reddit communities)
  These posts were selected to capture a broad spectrum of biased language, from explicit hate speech to subtle microaggressions.

### Annotation Framework

The SBIC dataset is **crowdsourced via Amazon Mechanical Turk**, using a **hierarchical annotation framework** that captures:

- **Offensiveness**: Whether the post is offensive (yes/maybe/no)
- **Intent to Offend**: Whether the author intended harm (yes/probably/probably not/no)
- **Lewdness**: Whether the post contains sexual content (yes/maybe/no)
- **Group Implication**: Whether the post targets a demographic group (yes/no)
- **Targeted Group**: Free-text annotation specifying the demographic group referenced
- **Implied Statement**: Free-text explanation of the biased stereotype implied
- **In-group Status**: Whether the author belongs to the referenced group (yes/maybe/no)

### Dataset Statistics

- **150,000 structured inference tuples**
- **34,000 unique group-implication pairs**
- **82.4% annotation agreement across tasks**
- **55% female, 42% male, 1% non-binary annotators**
- **Demographic skew: 82% White, 4% Asian, 4% Hispanic, 4% Black annotators**

### Inference Model for Social Bias Frames

#### Baseline Model Architecture

The study trains models to infer **Social Bias Frames** from text using **OpenAI's GPT and GPT-2** architectures. The task involves both **classification** and **text generation**, where models must:

- Predict categorical variables such as **offensiveness, intent, and group targeting**.
- Generate **implied social biases** in natural language.

#### Training and Evaluation

- **Models are fine-tuned on SBIC** using pre-trained transformer networks.
- **Hybrid classification + generation framework**:
- **Categorical variables** are learned via classification.
- **Implied bias statements** are generated in free text.
- Evaluation metrics:
  - Classification: **Precision, Recall, F1-score**
  - Text Generation: **BLEU, ROUGE-L, Word Mover’s Distance (WMD)**

#### Performance Analysis

- **80% F1-score for detecting offensive content**
- **High accuracy in identifying targeted groups**
- **Limited ability to generate nuanced bias explanations**
- **Models struggle with cases where bias is implied rather than explicit**


# Bias and Fairness in Large Language Models [19]

## Methods

The methodologies in this study provide a structured analysis of bias evaluation and mitigation techniques in large language models (LLMs). Bias evaluation metrics are categorized into embedding-based, probability-based, and generated text-based methods, each offering distinct insights but often producing inconsistent results. While embedding-based metrics, such as the Word Embedding Association Test (WEAT), assess bias within vector spaces, probability-based approaches like StereoSet and CrowS-Pairs measure bias through token likelihood differentials. Generated text-based metrics, on the other hand, analyze bias in model outputs but lack a standardized evaluation framework. Additionally, the study examines bias evaluation datasets, distinguishing between counterfactual input methods, which modify demographic terms to measure bias shifts, and prompt-based evaluations, which assess responses to structured queries. For bias mitigation, the study identifies four primary strategies—pre-processing, in-training, intra-processing, and post-processing techniques. Pre-processing methods, including data augmentation and reweighting, aim to balance representation in training data, while in-training modifications adjust learning objectives to prioritize fairness. Intra-processing and post-processing approaches, such as decoding strategy adjustments and fairness constraints, intervene at inference time to mitigate biased outputs. Despite these efforts, the study highlights the trade-offs between fairness and performance, with no single approach effectively eliminating bias without compromising generalization or accuracy.

Despite ongoing advancements in fairness research, several open challenges remain. There is no universal agreement on what constitutes "fair" AI, as different cultural, ethical, and legal perspectives lead to conflicting definitions of fairness. Furthermore, existing bias evaluation benchmarks are narrow in scope, primarily focusing on gender and racial biases while neglecting other dimensions such as socioeconomic disparities and regional linguistic diversity. Another limitation is that most bias mitigation techniques are evaluated in controlled settings rather than real-world applications. To create truly fair and unbiased AI, future research must adopt a more holistic approach that will consider context-aware bias detection, long-term fairness assessments, and more transparency in model development.


# Key Findings

## Measuring Language Ideologies in Text Data Selection
By replicating the GPT-3 in the quality filter and applying it to a dataset of high school newspaper articles, the researchers in [17] analyze which type of language are classified as "high quality" and what demographic and socioeconomic factors influence this classification. The results indicate that language quality filtering disproportionately favors text written by authors from wealthier, highly educated, urban backgrounds, highlighting systemic biases in how AI systems determine what constitutes "good" language. One of the most important findings of the study is that the GPT-3 quality filter does not align with traditional human assessments of high quality text. To test this, the authors compared the filter's classifications with factuality ratings for news sources, standardized test essay scores, and literary awards. 

**News Sources**: Articles from high and low factuality news outlets were rated similarly, indicating that the filter does not effectively distinguish reliable journalism from misinformation. 

**TOEFL Essays**: The model did not strongly correlate with human graders' assessments, suggesting that linguistic sophistication was not the primary determinant of quality scores. Instead, the model relied on factors such as document length and formality. 

**Pulitzer Prize Winning Books**: Here, the filter favored non-fiction and traditional literature, but rated poetry and drama significantly lower, reinforcing a bias towards certain writing styles. 

These discrepencies indicate that what AI considers "high quality text" is more reflective of formal, elite Western discourse, rather than linguistic accuracy or expressiveness.

Since AI language models are trained on text that has been filtered using these biased criteria, they inherit and perpetuate these linguistic preferences. The study highlights several concerning downstream effects:

  1. Language Homogeneity: AI models become less capable of understanding informal, regional, or dialectal variations of English.
  2. Reinforcement of Socioeconomic Biases: The exclusion of working-class, rural, and minority language styles results in models that are less inclusive and less representative of the full spectrum of human communication.
  3. Degradation of Model Fairness: AI models trained on these filtered datasets perform worse when engaging with diverse user populations, reinforcing structural inequalities in AI-driven decision-making.

The study ultimately calls for a paradigm shift in AI data selection practices, arguing that AI developers must critically examine how they define "quality" and ensure that these definitions do not perpetuate systemic biases. 

## Social Bias Frames 
While models in [18] achieve high accuracy (~80% F1-score) in identifying explicit bias and offensive language, their performance significantly declines when tasked with detecting implicit bias, identifying targeted demographic groups, and explaining the underlying stereotypes, with an F1-score dropping to ~66%. This suggests that while AI can recognize overtly discriminatory language, it struggles to infer the deeper, implied biases embedded in context. This demonstrates that detecting implicit bias remains a challenge as models frequently fail to infer unstated but implied stereotypes embedded in language.

For instance, when given the statement: _"We shouldn't lower our standards to higher more women"_, most models fail to correctly infer the implicit stereotype that women are less competent.
Additionally, generation-based evaluations show that AI struggles to generate contextually appropriate bias explanations.

The study also underscores the impact of training data composition, revealing that bias evaluation datasets predominantly contain gender, racial, and cultural stereotypes, yet are largely written in White-aligned English (78%), potentially skewing model performance across diverse linguistic styles. A major limitation uncovered in this study is AI's inability to generate meaningful, context-aware explanations for biased statements. Even when models correctly classify a statement as offensive, they often fail to provide accurate reasoning behind why it is biased. Models tend to rely more on word association rather than deeper contextual understanding.  Generated bias explanations are often repetitive or overly generic, rather than tailored to the specific stereotype implied in a statement. 

The study highlights several key challenges in improving AI fairness: 
  1. Models over-rely on explicit indicators of bias (i.e. offensive words) rather than pragmatic meaning and power dynamics
  2. Bias detection datasets lack representation of nuanced social biases such as economic class and disability.
  3. Bias explanation models need better commonsense reasoning capabilities to infer implicit stereotypes and historical contexts
To address these issues, future research should focus on expanding bias detection models to include intersectional biases and cultural variations.


## Bias and Fairness in Large Language Models: A Survey
The survey paper [19] highlights the inconsistencies in bias evaluation metrics, showing that different methods yield conflicting results, making fairness assessments unreliable. 

### Bias Evaluation Metrics: Inconsistencies and Limitations  
One of the biggest challenges in assessing bias in LLMs is the lack of consistency throughout evaluation metrics. Different bias assessment methods often produce conflicting results, making it difficult to determine the true extent of bias for any given model. 
Embedding-based metrics such as the Word Embedding Association Test (WEAT) focus more on word relationships within vector spaces, but fail to capture bias in context-dependent scenarios. Meanwhile, probability-based methods, such as StereoSet and CrowS-Pairs, measure token likelihood differentials to quantify bias in model output, but are highly sensitive to prompt phrasing and dataset composition. Generated-text evaluation, which analyzes AI-generated responses for bias, provides a more comprehensive perspective, but lacks a standardized framework for comparison across models. Due to these inconsistencies, there is no single, widely accepted benchmark for evaluation bias in NLP models, leading to fragmented results and unreliable fairness assessments. 

### Bias in Training Data: The Root of the Problem 
Much of the bias found in LLMs originates directly from their training data. These are often large-scale web-scraped datasets that reflect societal inequalities. Models that rely heavily on sources like Common Crawl, Wikipedia, and Books3 tend to inherit historical biases, gender stereotypes, and geographic disparities. The study highlights that quality filtering - a process that is used to remove noisy data - often reinforces existing biases rather than mitigating them. By prioritizing formally written, academic, or Western-centric text, quality filters disproportionately exclude informal dialects, community-based speech, and underrepresented cultural perspectives. This results in models that are less representative of global linguistic diversity, leading to biased outputs in downstream applications. Additionally, experiments comparing different datasets demonstrate that models trained on curated datasets exhibit lower bias, yet also perform worse on general NLP benchmarks. This indicates that there is a clear trade-off between fairness and task-specific performance. 

### Bias Mitigation Techniques: Trade-offs and Challenges 
Efforts to mitigate bias in LLMs generally fall into 3 categories: 
  1. **Pre-processing** (data-level interventions)
  2. **In-training** (modifying model learning objectives)
  3. **Post-processing** (adjusting outputs after inference)
While these techniques help reduce explicit biases, each comes with significant trade-offs. The study finds that none of these methods fully eliminate bias, instead each introduces competing priorities between performance, fairness, and real-world applicability. 

# Conclusion

In [17], Although no single action will solve this complicated issue, data curators and researchers could be more intentional
about curating text from underrepresented authors and groups, gathering sources from multiple genres and writing styles, and documenting their curation procedures and possible sources of exclusion. The study ultimately calls for a paradigm shift in AI data selection practices, arguing that AI developers must critically examine how they define "quality" and ensure that these definitions do not perpetuate systemic biases.

The study in [18] highlights several key challenges in improving AI fairness:

1. Models over-rely on explicit indicators of bias (i.e. offensive words) rather than pragmatic meaning and power dynamics
2. Bias detection datasets lack representation of nuanced social biases such as economic class and disability.
3. Bias explanation models need better commonsense reasoning capabilities to infer implicit stereotypes and historical contexts

To address these issues in [18], future research should focus on expanding bias detection models to include intersectional biases and cultural variations. It is also important to begin developing AI systems capable of reasoning about power hierarchies and unstated social implications. Improving dataset diversity will also help to ensure balanced representation across linguistic and demographic groups. Overall, this research underscores the gap between AI's ability to detect explicit bias versus its struggle to explain and contextualize bias in meaningful ways. Future advancements in pragmatic inference, commonsense reasoning, and bias-aware dataset construction are necessary to build more socially responsible AI models.

# Critical Analysis

The Research paper [17] brings up an important topic of underrepresentation of authors from different sections of society and selective favoritism of the few. The authors proved their point using a dataset which consisted of data from different locations across the US. They also mentioned and took critical care about data privacy since minors( High school students ) were involved which is highly appreciated. The paper however fails to mention in what cases would the language data previously been ignored be of an advantage to the others. Additionally, the study focuses mainly on high school newspapers and specific wordpress templates in the united states, which may limit its applicability to broader linguistic domains.

The Research paper[18] talks about issues not addressed popularly. We can always filter out words, offensive language but what if what is implied is not directly equal to what is said. This tricky language structuring can get past the traditional filtering algorithms and thus the paper introduces Social Bias Frames to detect languages that aim to model the pragmatic frames in which people project social biases and stereotypes onto others. This framework is particularly also useful for understanding how language can reinforce stereotypes and power imbalances. However, the annotator pool is limited to the US and Canada which can introduce some amount of bias in terms of different views. Also working on offensive language can be disturbing but is like a bitter pill to bite but better measures could have been put in place like limiting the number of annotations, better compensations and regularly monitoring their mental health.

The Research paper [19] provides a comprehensive taxonomy of bias evaluation, mitigation techniques, and fairness-tradeoffs. The key strength of the survey lies in its extensive coverage of bias evaluation and mitigation techniques, along with its well-structured taxonomy of metrics, datasets, and methodologies. Additionally, it provides a critical assessment of the limitations associated with each approach and highlights the need for further research to deepen the understanding of bias, fairness, and the social groups involved. The research presented in the survey may primarily represent the perspectives of dominant groups. The classification of social groups in these studies should be scrutinized, as the very act of defining them can reinforce social constructs, uphold power imbalances, and sustain systems of oppression.

# References

[17]. Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection. S. Gururangan et al., 2022
[18]. Social Bias Frames: Reasoning about Social and Power Implications of Language. M. Sap, S. Gabriel, L. Qin, D. Jurafsky, N.A. Smith, Y. Choi, 2020.
[19]. Bias and Fairness in Large Language Models: A Survey. I.O. Gallegos et al. 2023
