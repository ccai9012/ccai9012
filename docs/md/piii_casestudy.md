
# A.1.3: Case Study Topic Area #3: Datasets & Risks

## _The Building Blocks, Blind Spots & Consequences_

<div style="height:1.5rem"></div>

_Note 1: This outline provides guidance specific to Case Study #3, complementing the **general case study rubric**_ [**[link]**](casestudy.html).  
_Note 2: See lecture note 3.1 **Datasets & Risks of AI** [**[link]**](piii_data_and_risks.html) for illustrative examples._

<div style="height:4rem"></div> 

<hr style="height: 6px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## The Brief

<div style="height:1.5rem"></div> 

Generative AI systems learn from data. What they can generate, recognise, or reason about is therefore shaped by **what is represented in that data, what is missing, and how the data has been collected and curated**.

In this case study, you will investigate:

**Data → Model Behaviour → Risk → Response**

Select a **specific domain or task** and use a market-ready Generative AI model or LLM to explore:

- What the model does well
- Where its capability begins to break down
- What these patterns might reveal about its underlying data
- What risks arise from these limitations
- What could be improved

You are **NOT training a customised Generative AI model.**

You are using controlled tests, comparisons, and available information about datasets to **probe the boundaries of an existing AI system**.

<div style="height:1.0rem"></div>

**Examples:**

• Ask a Generative AI model to transform **daytime Hong Kong street images into realistic nighttime scenes**, and investigate where local characteristics are preserved, distorted, or replaced by generic imagery.

• Test whether an LLM performs equally well across **different locations, languages, cultures, disciplines, or types of knowledge**.

• Identify **another specific task in your own domain** where you suspect the model's performance may depend on what is well or poorly represented in its training data.

The goal is not simply to show that AI makes mistakes.

The goal is to investigate **what those mistakes or boundaries may reveal about the data behind the system, and why this matters**.

<div style="height:1.0rem"></div>

Your analysis should address the following questions.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### 1. What is the domain-specific task?

<div style="height:1.0rem"></div>

Define a **specific and testable task**.

(E.g., creating a realistic nighttime Hong Kong street scene from a daytime photograph, rather than simply generating an attractive nighttime image.)

<div style="height:1.0rem"></div>

• **Problem / Function**

&nbsp;&nbsp;&nbsp;&nbsp;– What is the **main objective** of the task?  
&nbsp;&nbsp;&nbsp;&nbsp;– What would count as a successful result?  
&nbsp;&nbsp;&nbsp;&nbsp;– Why is this problem important?  
&nbsp;&nbsp;&nbsp;&nbsp;– Why might the available data matter?

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 2. What data might shape the model's behaviour?

<div style="height:1.0rem"></div>

Investigate what is known about the data behind the selected AI system and identify the aspects most relevant to your task.

• **Data & Representation**

&nbsp;&nbsp;&nbsp;&nbsp;– What is publicly known about the model's training data?  
&nbsp;&nbsp;&nbsp;&nbsp;– Where might the data have come from?  
&nbsp;&nbsp;&nbsp;&nbsp;– What kinds of examples are likely to be well represented?  
&nbsp;&nbsp;&nbsp;&nbsp;– What might be missing or underrepresented?

• **Data Quality**

Consider where relevant:

&nbsp;&nbsp;&nbsp;&nbsp;– Coverage and diversity  
&nbsp;&nbsp;&nbsp;&nbsp;– Accuracy and labelling  
&nbsp;&nbsp;&nbsp;&nbsp;– Geographic or cultural representation  
&nbsp;&nbsp;&nbsp;&nbsp;– Recency  
&nbsp;&nbsp;&nbsp;&nbsp;– Filtering and curation  
&nbsp;&nbsp;&nbsp;&nbsp;– Privacy, consent, copyright, or provenance

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 3. What does the model know, miss, or distort?

<div style="height:1.0rem"></div>

Systematically test the model on your chosen task.

Do not rely on only one successful and one unsuccessful example. Look for **repeatable patterns**.

• **Capabilities**

&nbsp;&nbsp;&nbsp;&nbsp;– What does the model consistently handle well?  
&nbsp;&nbsp;&nbsp;&nbsp;– What knowledge or patterns appear to be well represented?

• **Boundaries**

&nbsp;&nbsp;&nbsp;&nbsp;– Where does the model begin to fail?  
&nbsp;&nbsp;&nbsp;&nbsp;– What does it omit, simplify, substitute, hallucinate, or misrepresent?  
&nbsp;&nbsp;&nbsp;&nbsp;– Are these failures random or systematic?

• **Evidence**

You may use:

&nbsp;&nbsp;&nbsp;&nbsp;– Controlled prompt changes  
&nbsp;&nbsp;&nbsp;&nbsp;– Controlled input changes  
&nbsp;&nbsp;&nbsp;&nbsp;– Repeated generations  
&nbsp;&nbsp;&nbsp;&nbsp;– Comparisons across categories or conditions  
&nbsp;&nbsp;&nbsp;&nbsp;– Comparisons between different models  
&nbsp;&nbsp;&nbsp;&nbsp;– Real-world or documented reference material

Prompts may be used as an **experimental probe**, but the goal is not simply to demonstrate better prompt engineering.

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 4. What risk does this create?

<div style="height:1.0rem"></div>

Connect the behaviour you observed to at least **one meaningful downstream risk**.

Possible risks include:

- Representational bias
- Missing or unreliable knowledge
- Stereotyping
- Geographic or cultural misrepresentation
- Privacy or consent
- Copyright or provenance
- Unsafe or misleading outputs
- Exclusion of underrepresented users or contexts

<div style="height:1.0rem"></div>

• **Consequences**

&nbsp;&nbsp;&nbsp;&nbsp;– Who or what could be affected?  
&nbsp;&nbsp;&nbsp;&nbsp;– When does the limitation become consequential?  
&nbsp;&nbsp;&nbsp;&nbsp;– Could it influence judgement, representation, access, or decision-making?

Your analysis should connect a **specific model behaviour** to a **specific consequence**.

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 5. How might it be improved?

<div style="height:1.0rem"></div>

Propose a reasonable response to the limitation you identified.

Possible approaches include:

- Adding more diverse or representative data
- Improving labels or metadata
- Introducing specialised domain data
- Improving geographic, cultural, or linguistic coverage
- Filtering poor-quality or problematic data
- Retrieving information from trusted external sources
- Fine-tuning or adapting a model
- Introducing human review or verification
- Using prompts or guardrails where the underlying model cannot be changed

<div style="height:1.0rem"></div>

• **Causal Reasoning**

&nbsp;&nbsp;&nbsp;&nbsp;– What would you change?  
&nbsp;&nbsp;&nbsp;&nbsp;– Why might this improve the behaviour you observed?  
&nbsp;&nbsp;&nbsp;&nbsp;– What trade-offs or new risks might it introduce?

Your analysis should reveal **one clear insight about how data shapes model behaviour and risk**.

<div style="height:1.0rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### What a Strong Submission Demonstrates

<div style="height:1.0rem"></div>

A strong submission will:

- Define a clear **domain-specific task**
- Identify a meaningful issue related to **data quality, coverage, or representation**
- Use evidence to reveal a **capability, bias, gap, or boundary**
- Distinguish clearly between **what is known, observed, and inferred**
- Connect model behaviour to a meaningful **risk or consequence**
- Propose a plausible **response or improvement**
- Communicate the relationship clearly:

**Data → Behaviour → Risk → Response**

Your work should demonstrate **data-centric AI thinking**, not simply better use of an AI tool.

<div style="height:4.0rem"></div> 

<hr style="height: 6px; border: none; background-color: #000;">

<div style="height:2.0rem"></div> 

## Deliverables

<div style="height:1.5rem"></div>

Your submission is built around a central **artefact**, which forms the backbone of the case study and provides the evidence from which your abstract, slides, video, and poster are constructed.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### The Artefact — Data, Behaviour & Risk

<div style="height:1.0rem"></div>

The artefact is the **core analytical and evidentiary material** developed through your investigation.

Think of it as the body of evidence that makes the relationship between **data, model behaviour, risk, and response** visible.

Your artefact should make clear:

• **Task & Context**

&nbsp;&nbsp;&nbsp;&nbsp;– What specific task are you testing?  
&nbsp;&nbsp;&nbsp;&nbsp;– What would count as successful or credible performance?

• **Data & Representation**

&nbsp;&nbsp;&nbsp;&nbsp;– What is known about the relevant training data or data pipeline?  
&nbsp;&nbsp;&nbsp;&nbsp;– What kinds of information appear well represented, missing, or underrepresented?  
&nbsp;&nbsp;&nbsp;&nbsp;– Where necessary, distinguish clearly between **what is known and what is inferred**.

• **Testing & Evidence**

&nbsp;&nbsp;&nbsp;&nbsp;– Use controlled tests, comparisons, repeated outputs, or reference material to reveal patterns in model behaviour.  
&nbsp;&nbsp;&nbsp;&nbsp;– Show both **capabilities and boundaries**, rather than relying on isolated examples.

• **Risk & Consequence**

&nbsp;&nbsp;&nbsp;&nbsp;– Connect an observed behaviour to a meaningful downstream risk.  
&nbsp;&nbsp;&nbsp;&nbsp;– Explain **who or what may be affected, and why the behaviour matters**.

• **Response**

&nbsp;&nbsp;&nbsp;&nbsp;– Propose a plausible intervention addressing the limitation you identified.  
&nbsp;&nbsp;&nbsp;&nbsp;– Explain why the intervention might improve the behaviour, and what trade-offs it may introduce.

<div style="height:1.0rem"></div>

Use **evidence from your investigation** to support these claims.

This may include:

• Input examples  
• Prompts and outputs  
• Controlled comparisons  
• Repeated generations  
• Dataset documentation  
• Screenshots  
• Diagrams  
• Tables or quantitative observations  
• Real-world reference material  
• Model comparisons  
• Annotations identifying specific patterns or failures  

Each piece of evidence should be carefully selected and **captioned to explain what it shows, what you infer from it, and why it matters**.

The artefact should allow the reader to follow the central chain:

**What data? → What did you test? → What happened? → What risk does it create? → What could change?**

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### i. Abstract

<div style="height:1.0rem"></div>

Provide a concise written abstraction of **no more than 300 words**.

Distil the task investigated, the relevant data issue, your method of testing, the key behavioural finding, the resulting risk, and your principal insight about **how data shapes AI behaviour**.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### ii. Slides

<div style="height:1.0rem"></div>

Develop a sequence of **portrait-format slides** that uses your artefact and supporting evidence to construct a clear visual story.

The sequence should make the progression from **data → testing → behaviour → risk → response** legible.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### iii. Video

<div style="height:1.0rem"></div>

Develop a **short-form video presentation of no more than 90 seconds** using your slides and artefact as its visual foundation.

Use the video to connect the strongest evidence and communicate your central insight about **how data shapes what the AI system can, cannot, or should not be trusted to do**.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### iv. Poster

<div style="height:1.0rem"></div>

Bring the strongest elements of your artefact and evidence together into a **complete A2-format poster**.

The poster should function as a **self-contained visual argument**, making the relevant data issue, observed model behaviour, resulting risk, and principal insight clearly legible.

<div style="height:1.0rem"></div>

You may include **supplementary materials in an appendix**, such as:

- Additional test cases
- Prompt comparisons
- Model comparisons
- Dataset documentation
- Code snippets
- Additional diagrams or tables
- Experimental results

<div style="height:4rem"></div> 

<hr style="height: 6px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## Grading Criteria

<div style="height:1.0rem"></div>

### In a nutshell

#### We are **not** grading:

- Advanced mathematics
- Sophisticated coding
- The complexity of the AI tool
- The aesthetic quality of AI outputs in isolation
- Complete knowledge of a proprietary training dataset

<div style="height:1.5rem"></div>

#### We **are** grading:

| Criterion | What It Means |
|-----------|---------------|
| **Data Understanding** | Do you understand how data quality, coverage, or representation may shape the model? |
| **Evidence & Reasoning** | Do you support your argument with observable evidence and distinguish evidence from speculation? |
| **Risk Analysis** | Do you explain why the behaviour matters and what consequences may follow? |
| **Response** | Do you propose a plausible way to address the identified issue? |
| **Communication** | Can you construct a clear and visually coherent argument? |

<div style="height:1.5rem"></div>

### Rubric

| Criterion | Excellent | Adequate | Insufficient |
|-----------|-----------|----------|--------------|
| **Data Understanding** | Clearly identifies and explains a meaningful issue in data quality, coverage, representation, or provenance | Identifies a relevant data issue but with limited depth | Data is treated superficially or its relevance is unclear |
| **Evidence & Reasoning** | Uses systematic evidence to connect data-related factors to model behaviour while distinguishing observation from inference | Provides some evidence but the connection is underdeveloped | Makes unsupported claims or relies on isolated outputs |
| **Risk Analysis** | Clearly connects observed behaviour to a specific and meaningful downstream risk | Identifies a relevant risk but connection is weak | Mentions generic risks without linking them to the investigation |
| **Response** | Proposes a convincing response and explains why it could address the identified issue | Suggests a plausible response with limited justification | Suggests generic fixes without addressing the underlying issue |
| **Clarity & Visual Argument** | Evidence is carefully selected and organised into a clear visual narrative | Argument is understandable but uneven | Submission is difficult to follow or visually disconnected |

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## The Standard

<div style="height:1.0rem"></div>

The goal is not simply to show that **AI sometimes gets things wrong**.

The goal is to ask:

**What does the system appear to know?**  
**What is missing or distorted?**  
**What might the data have to do with this?**  
**What risk does this create?**

Distinguish **what you know, what you observe, and what you infer**.

Don't just use AI.

**Probe what it has learned—and what it has not.**

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

