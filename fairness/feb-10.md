# Fairness - Intro & Bias Sources 

< general introduction >

## [Fairness and Machine Learning, Ch 1](https://fairmlbook.org/introduction.html). S. Barocas, M. Hardt, A. Narayanan, 2023

**Introduction** 

Historically, we have used statistics to help us in decision-making, to more accurately predict outcomes. Machine learning builds upon the idea of using statistical measures to help us make decisions, but instead of manually selecting features and calibrating weights by hand,is able to uncover patterns and relationships in data on its own. It has its own flaws, however. To have a good model, you need good data—because a model generalizes outputs based off of its inputs.If data reflects prejudices based on stereotypes or inequalities, the model will also reflect them. The model may also reflect errors in human judgement—for example, younger defendants are statistically more likely to re-offend, but judges are more likely to be less harsh when deciding on their sentences due to the belief that they deserve another chance since they are young


**Demographic Disparities** 

Bias due to race is not new in data-driven systems. For example, Amazon uses such a system to decide where they can complete free same-day deliveries—however, neighborhoods that qualified for this perk were twice as likely to be home to white residents than black residents. Even though Amazon says that the decision was driven by efficiency and costs, and not race, this racial disparity could be due to historical inequalities and segregation.

Going forward, how do we define bias in model decisions?

- Bias: demographic disparities in algorithmic systems that are objectionable for societal reasons. 
- Statistical Bias: when expected or average values differ from the true values it aims to estimate. 

Both biases should be considered when developing machine learning models. 


**Machine Learning Loop** 

![ML Loop](images/machine-learning-loop.png)

How do disparities propagate themselves through the machine learning model process? 

First, measurements are collected, which the model will be trained on. Issues in these values—such as patterns of stereotypes or disproportional representation—can later affect a model trained on the data. Then, a model goes through a learning stage, where we train it on the collected measurements. Afterwards, a model is able to predict outputs. These outputs will be representative of the general patterns learned from the original collected data. Some models continue to learn through feedback, potentially from users—although this can be used to unlearn biases, it may also be another way for bias to arise.


**The State of Society** 

Disparities in people due to gender, socioeconomic factors, discrimination, and others reflect themselves in training data, thus later effecting model outcomes.

What are some examples of this? 

- Bureau of Labor Statistics (2017): some job occupations have stark gender imbalances. Machine learning systems that screen for job candidates might learn this gender division, and discriminate because of it. 
- Street Bump: this tool, designed to automatically collect data on potholes using smartphones, reflecting patterns of phone ownerships. This meant that wealthier neighborhoods had better coverage than lower-income ones, or places with majority elderly populations, who were less likely to own smartphones. 
- Kaggle: Automated Essay Scoring datasets can contain biases from human graders for student essays, potentially from linguistic choices that reflect social groups—which is a pattern that models learned when trained on. 


**The Trouble with Measurement** 

The problem with measurement is that it requires defining variables of interest. Biases in definitions, categories, and how we quantifiably measure qualitative metrics such as success come into play. Current social norms are reflected in this, and certain variables have to be reduced to a single number. For example; 

- A good employee might be reduced to performance review scores. 
- A succesful student might be reduced to their GPA. 

Measurements also change over time. Racial categories have evolved: in 2008, "multiracial" became an option on forms. Over time, we've developed new gender labels. These measurements are only available in newer datasets, and not older ones. 


**From Data to Models** 

Models extract stereotypes represented in data as much as they extract knowledge of bigger pictures we want them to learn. Sometimes, removing features such as gender to remove bias is not enough, due to proxies and other feature correlations with gender. Machine learning algorithms generalize based on the majority culture: which causes issues and errors to more likely occur with minority groups. 

![ML Loop](images/gender-stereotypes.png)

For example, the figure above shows how a model learned the association between gender and occupations: Turkish has gender neutral pronouns, but automatically determined the doctor to be a "he" while the nurse was deteremiend to be a "she". 


**The Pitfalls of Action** 

We have to pay attention to where models are applied: population characteristics change over time, and different populations may have different cultural / social norms. Models need to reflect these changes! Models are also limited to observing correlations, not necessarily causations—and understanding why models make decisions is important. This is not always easy due to the black-box nature of some algorithms. Model predictions can also affect outcomes that will in turn invalidate its own predictions—if a model observes that there is less traffic using a certain path, and recommends it, that path will receive more traffic due to the suggestion, and may end up with more traffic than other routes. 


**Feedback and Feedback Loops** 
 
User feedback can be used to refine models. However, feedback can be misinterpreted, and may also contian user prejudices. Feedback can occur in three main ways: 

- *Self-fulfilling prophecies*: this is esentially confirmation bias. When predicting a certain outcome, this results in looking for that particular outcome, validating itself in the process (without looking at other factors). An example of this is a policing system: knowing that crime occurs more often in a certain neighborhood, more police officers will be sent to that location, which will in turn lead to more arrests and reports of crime—leading to an infinite, self-feeding cycle. 

- *Predictions that affect the training set*: due to self-fulfillment, using model predictions to update a model leads to a feedback loop with bias reinforcing existing bias in a model. Models should only be updated by surprising or new outcomes, since refining it with duplicate data will just cause existing bias to be more solidified. 

- *Predictions that affect the phenomenon and society at large*: long-standing prejudices will be reflected in models, which will keep prejudices alive, which will later reinforce prejudices in models. This again leads to a never-ending cycle of prejudice perception. 


**Getting Concrete with a Toy Example** 

![toy_example](images/toy-example.png)

The paper references a made-up example where a classifier is trained to predict job performance based on college GPA and an interview score, and determines a cutoff for hiring interviewees. Notice how the cutoff favors triangles over squares, even though it does not take into account whether a person was of the triangle or square group to make its prediction. This is an example of how a demographic group may be represented by proxies in the data—even if gender or race was not explicitly considered, performance scores may reflect manager biases, for example. The point is that removing features that we do not want a model to learn patterns for is not enough, due to proxies and other correlations between data fields. 


**Justice Beyond Fair Decision Making** 

- Interventions that Target Underlying Inequities: instead of trying to optimize selections and decisions with a model, we should target the reasons why there might be disparities. For example, instead of judging an employee solely on their performance review score, there should also be a focus on building environment, accessibility, etc, to give everyone better opportunities to be on a level playing field.  

- The Harms of Information Systems (search and recommendation algorithms) 
    - Allocative Harms: when a system withholds groups opportunities / resources. 
    - Representational Harms: when a system reinforces subordination of groups along the lines of identity. This receives less attention due to non-immediate harm caused, but they have long-term effects on culture and stereotypes. 


**Limitations & Opportunities** 

Completely unbiased measurements may be infeasible, but we still have to try our best to remove bias where we can. Observational data can be insufficient to identify disparity causes, however, due to lack of understanding of models, but this is necessary to intervene and remedy biases. Ensuring fairness makes model decisions more transparent, forcing the articulation of decision goals—but this requires making models explainable, which can be difficult.


**Critical Analysis**

The paper was overall vague: which is good as an introductory chapter, but lacked substance. There were a few good examples to illustrate points, but the separate sections seemed to cover the same points a few times, making the content repetitive.


## [Big Data: A Report on Algorithmic Systems, Opportunity, and Civil Rights](https://obamawhitehouse.archives.gov/sites/default/files/microsites/ostp/2016_0504_data_discrimination.pdf). The White House, 2016

## [Big Data’s Disparate Impact](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899). S. Barocas, A. Selbst, 2014

### Introduction and Motivations

While we could trace back human decisions to their source, and hold them directly responsible, can the same be said about an algorithm? If an automated decision is made by an algorithm (say with picking a candidate for a job), who can be held responsible for any discrimination such a decision might have caused?

It is well known that algorithms rely on data - mined from different sources - to make their decisions. Even though efforts can be made to prevent discrimination, when the dataset itself is biased (wanton or not) they tend to propagate to downstream tasks in computer algorithms. This can lead to an unfortunate result of exacerbating already prevalent inequalities which marginalize certain groups. 

The 14th amendment of the US constitution offers the Equal Protection Clause, which led to the passing of the Civil Rights Act of 1964 and its component Title VII - Prohibition of Discrimination in Employment based on race, color, religion, sex, and national origin. This paper explores the idea of fairness and the impact of automated decision making tools through this lens – Title VII can be applied towards remedying some, if not all, of these effects. Since it is backed by law, victims can approach a court to seek redress from discriminatory decisions.

Granted, discriminatory practices are far more widespread and far reaching than employment, the authors of this paper ground their arguments to Title VII given its “particularly well-developed set of case law and scholarship”. They present this as a literature review of academic writing, current events (as of 2014), and case studies of court rulings. While agreeing that the current framework is insufficient to fully tackle the scope of discrimination given the growing use of algorithms, they are motivated by this fact to address the issue. They posit that discrimination can be an artifact of the data mining process itself even without any human intervention. 


### Methods

Over three parts, the authors adopt an interdisciplinary approach, drawing on computer science literature, legal scholarship, and antidiscrimination law.

* The essay first presents an overview of discriminatory mechanisms within the data mining process:
    * Defining target variables and class labels, and how choices in specifying the problem can unintentionally put protected classes at a disadvantage
    * Feature selection within (training) data which can amplify existing biases within the data (“garbage in, garbage out”). They also discuss proxies, where seemingly neutral features can serve to identify membership in protected classes.
* The authors review Title VII jurisprudence, analyzing the disparate treatment & disparate impact theories, and how these doctrines apply to various mechanisms of data mining.
* They also analyze case law and Equal Employment Opportunity Commission (EEOC) guidelines, discussing how courts determine if a practice is justified as a “business necessity” as it relates to data mining.
* They consider both inherent (arising from the mechanisms of data mining) and external (political/legal) difficulties of reforms in ensuring fairness for all.


### Key Findings

* Data mining can lead to discrimination even without intentional bias, since it could be an emergent property of data mining.
* Mechanisms within data mining can lead to discrimination, including how a target variable is defined, how training data is collected and labelled, feature selection, and the use of proxies/masking.
* While data scientists designing algorithms find the greatest risk in false negatives/positives, a greater risk lies in the conceptual accuracy of the class labels, such as the rating an algorithm would give an employee, and the variables considered, such as prior performance reviews.
* Annotating samples is difficult ethically to do by hand while ensuring fairness.
* “Garbage in, garbage out”: as also discussed earlier, algorithms will reflect bias in the data they mine. An example is the LinkedIn talent match feature which recommends candidates to employers based on demonstrated interest in certain candidate properties. This algorithm reflects the social discrimination biases of employers when data mining.
* Privileged people are better documented and have more available data, boosting their rankings in automated job searches
* Under Title VII, an employer can get sued for discrimination under two theories: disparate treatment and disparate impact. Treatment is intentional and impact is from a neutral policy that effectively discriminates. As discriminatory data mining is unintentional it is difficult to argue for treatment. To argue for impact, a plaintiff must find an alternative employment practice that is reasonable and bias-free. Both these tend to be inadequate in their scope while considering data mining.
* Job hiring is inherently subjective so it is difficult to draw a line between discrimination and fair hiring if there is no clear intent to discriminate.
* Reforms in the field face both internal and external challenges, making it difficult to modify anti-discrimination laws to truly benefit protected classes. 


### Critical Analysis.

**Strengths**

The essay provides a detailed account of how data mining can lead to discrimination
Offers a useful taxonomy of the specific mechanisms within data mining that can generate unfair decisions, and is able to effectively tie CS literature detailing algorithmic bias with real life cases of its impacts on employment discrimination.
The paper details the history of employment discrimination court cases and how they may apply to modern examples of discriminatory data mining when used for job hiring.
They highlight the limitations of current legal frameworks (particularly Title VII) in addressing the challenges of data mining


**Weaknesses**

* While it discusses the limitations of current laws, it offers few specific recommendations for reform.
* As the authors also agree, some, if not most, instances of discriminatory data mining will not generate liability under Title VII.  This is a feature of the current approach to anti-discrimination jurisprudence, which focuses on procedural fairness. 
* The paper is limited to Title VII, and does not fully explore other areas of anti-discrimination law even though it claims the findings apply elsewhere as well. It also limits itself to US law (possibly by design), and could benefit from a discussion of current best practices in other countries.
* Perhaps minor, but the paper includes a lot of jargon which could make it difficult for both sides (technical & legal) to completely understand the ideas presented. Due to the intricate nature of the policies discussed, the paper ends up lengthy and difficult to quickly understand the major points of.

**Potential Biases**

As this paper is structured as a literature review, the selection of literature, court cases, and events discussed could potentially introduce bias. Given that the paper is almost a decade old, the emergence of more equitable practices in the past few years could render some of the arguments invalid.

**Ethical Considerations**

Similar to the first paper, this paper brings up points about where discrimination occurs in models and data, while discussing existing laws, legal topics, concerning data—explains their insufficiency. 



## [Semantics derived automatically from language corpora contain human-like biases](https://www.science.org/doi/10.1126/science.aal4230). A. Caliskan, J.J. Bryson, A. Narayanan, 2017

### Introduction and Motivations 

The researchers set out to study how machine learning models trained on large language corpora learn and reflect human biases. Existing works have observed that models learn racial stereotypes based on names. Furthermore, female names are more associated with family than career words, compared to male names. Other works have found that other, more general human biases can be observed: for example, flowers have higher correlations with being pleasant, while the word insects is more closely tied to the word unpleasant. Using GloVe word embeddings, the researchers demonstrate that models also capture stereotypes related to gender, regarding occupation and names.


### Methods

The researchers develop two methods: a Word-Embedding Association Test (WEAT, to document human biases) and Word-Embedding Factual Association Test (WEFAT, to see how word embeddings correlate with real-world statistics). TODO, WORK IN PROGRESS


### Key Findings

![Gender_Association_Figures](images/gender-association.png)

The findings show that models trained on texts learned gender biases embedded in human language data—this is concerning, because these biases can affect real-world applications such as hiring algorithms or resume screenings. TODO, WORK IN PROGRESS


### Critical Analysis

TODO, WORK IN PROGRESS 

