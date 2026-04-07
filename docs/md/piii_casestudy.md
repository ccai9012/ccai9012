# A.1.3: Case Study #3: Datasets & Risks

## _The Building Blocks & Guardrails_

<div style="height:1.5rem"></div>

_Note 1: This outline provides guidance specific to Case Study #3, complementing the **general case study rubric**_ [**[link]**](casestudy.html).  
_Note 2: See lecture note 3.1 **Datasets & Risks of AI** [**[link]**](piii_data_and_risks.html) for some illustrative examples of the tasks._

<div style="height:4rem"></div> 

<hr style="height: 6px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## The Brief

<div style="height:1.5rem"></div> 

Understand how the training dataset guides the **capability and boundary** of a Generative AI model.

Investigate how specific training datasets can differentiate the capability of Generative AI (GenAI) models or LLMs.

You are **NOT training customized Generative AI models.**  
You are **exploring the boundaries of market-ready GenAI/LLMs in specific tasks** (domain knowledge).

<div style="height:1.0rem"></div>

**Examples:**

• It can be asking a pre-trained LLM to generate **"realistic" nighttime images** based on given input daytime images (download daytime images from the [**[link]**](https://drive.google.com/drive/folders/1m0xVRvCSXI9uI1zXKCypMNZ5-kXnNF4-?usp=sharing ))

• Or you can think about **another specific task in your domain of study** that you notice current LLMs on the market cannot handle well, but through very specific prompts or better datasets, they might improve their capabilities.

Your analysis should address the following questions.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### 1. What is the domain-specific task?

(E.g., creating realistic nighttime street view images from daytime ones instead of generating a fancy rendering or fantasy-like visions)

<div style="height:1.0rem"></div>

• **Problem / Function**

&nbsp;&nbsp;&nbsp;&nbsp;– What is the **main objective** of the task?  
&nbsp;&nbsp;&nbsp;&nbsp;– Why is this problem important?  
&nbsp;&nbsp;&nbsp;&nbsp;– Why LLM (or GenAI) might be useful?

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 2. What models do you use for the task?

(E.g., Do you generate the nighttime images with StableDiffusion, Nano Banana, Midjourney, etc.?)

<div style="height:1.0rem"></div>

• **Model Dependency**

&nbsp;&nbsp;&nbsp;&nbsp;– Describe clearly how it **operates**.  
&nbsp;&nbsp;&nbsp;&nbsp;– Explain how the overall task breaks down into **input data preparation, prompt (or algorithmic steps), and output steps**.

• **Your explanation should show**

&nbsp;&nbsp;&nbsp;&nbsp;– The **overall workflow**  
&nbsp;&nbsp;&nbsp;&nbsp;– The **specific model's mechanism** you are leveraging (E.g., if you are using StableDiffusion, explain how the Diffusion architecture works)  
&nbsp;&nbsp;&nbsp;&nbsp;– The full **Inputs → transformations → outputs link**

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 3. Is the GenAI model (or LLM) powerful in fulfilling the domain-specific task?

(E.g., mimicking a realistic nighttime urban environment?)

<div style="height:1.0rem"></div>

• **Capabilities**

&nbsp;&nbsp;&nbsp;&nbsp;– What strengths or behaviors does the model enable?

• **Limitations**

&nbsp;&nbsp;&nbsp;&nbsp;– What problems or dysfunctionality do you spot/diagnose from the pre-trained model?

<div style="height:0.5rem"></div>

---

<div style="height:1.0rem"></div>

### 4. How do you improve it?

<div style="height:1.0rem"></div>

• **Causal reasoning**

&nbsp;&nbsp;&nbsp;&nbsp;– Explain how training dataset quality, and/or prompt engineering might (or might not) improve the model's ability for the domain-specific task.

• **Comparison**

&nbsp;&nbsp;&nbsp;&nbsp;– Where helpful, compare the **"initial results"** to an **"improved result"** from an alternative model, or an improved prompt, or feeding in specific training data.

Your analysis should reveal **one key insight** about how the domain-specific training dataset/prompt/harness reshapes model performance.

<div style="height:1.0rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### What a Strong Submission Demonstrates

<div style="height:1.0rem"></div>

A strong submission will:

- Isolate the role of the **training dataset quality** from the mechanism clearly  
- Diagram the **overall workflow of the model**  
- Zoom into the **selected module** (training dataset/prompt/etc)  
- Identify **inputs → transformations → outputs**  
- Explain how the **training dataset matters to behavioral outcomes**

Your writing should demonstrate **training dataset engineering thinking or prompt engineering thinking**.

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## Deliverables

<div style="height:1.5rem"></div>

Your submission must include **four parts**.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### i. Annotation (≈500 words)

<div style="height:1.0rem"></div>

Provide a short written annotation accompanying your case study.

• **Title**  
&nbsp;&nbsp;&nbsp;&nbsp;– A clear title identifying the system or mechanism studied.

• **Caption / Description**  
&nbsp;&nbsp;&nbsp;&nbsp;– A brief written explanation summarising the focus of your case study.

<div style="height:0.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.0rem"></div>

### ii. The Artefact

<div style="height:1.0rem"></div>

Your artefact consists of **two components**:

• **Supporting Explanation** — contextual material that frames the domain specific task you explore  
• **Input → Model → Output Representation** — the core explanation showing how the whole input-output flow operates.

<div style="height:1.0rem"></div>

---

<div style="height:1.0rem"></div>

### iii. Vignette

<div style="height:1.0rem"></div>

The vignette is a **short-form video presentation** summarising your case study.

• Duration: **approximately 60 seconds**

Your video should clearly communicate:

• The **GenAI or LLM model studied**  
• The **domain-specific task you investigated**  
• Your **key insight about how the training dataset or prompt reshapes AI model behaviour**

<div style="height:1.0rem"></div>

---

<div style="height:1.0rem"></div>

### iv. Evidences (Optional)

<div style="height:1.0rem"></div>

You may include **supplementary materials in an appendix**, such as:

• Code snippets  
• Parameter sweeps  
• Prompt comparisons  
• Additional diagrams or tables  
• Experimental results

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## Grading Criteria

<div style="height:1.0rem"></div>

### In a nutshell

**We are not grading:**

- Advanced mathematics  
- Sophisticated coding  
- Exhaustive technical coverage

**We are grading:**

| Criterion | What It Means |
|-----------|---------------|
| **Understanding** | Do you correctly explain how the mechanism works? |
| **Causal Reasoning / Comparison** | Do you link structural design choices to observable system behaviour, including how behaviour might change under alternative designs? |
| **Limitations** | Do you identify meaningful trade-offs, constraints, or failure modes introduced by the mechanism? |
| **Clarity** | Can you explain complex ideas in a precise and accessible manner? |
| **Visual Explanation** | Do your diagrams or visual representations clearly communicate the mechanism and information flow? |

### Rubric

| Criterion | Excellent | Adequate | Insufficient |
|-----------|-----------|----------|--------------|
| **Mechanism Accuracy** | Structurally correct and clearly articulated explanation of how the mechanism operates | Mostly correct explanation with minor gaps or simplifications | Superficial, vague, or incorrect description of the mechanism |
| **Causal Analysis** | Clear linkage between structural design and system behaviour, with thoughtful comparison or reasoning | Some connection between structure and behaviour, but underdeveloped | Describes outputs or observations without structural reasoning |
| **Failure Awareness** | Identifies meaningful trade-offs, limitations, or structural constraints | Mentions limitations but without depth or explanation | No meaningful discussion of limitations |
| **Clarity of Communication** | Complex mechanism explained clearly and accessibly to non-specialists | Explanation understandable but dense or uneven | Obscure, overly technical, or difficult to follow |
| **Visual Explanation** | Diagrams clearly reveal the system workflow, mechanism, and information flow | Visuals present but only partially clarify the mechanism | Visuals absent, confusing, or decorative rather than explanatory |

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## Submission Deadline

<div style="height:1.0rem"></div>

Original: **2026.04.29 00:00**

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

## The Standard

<div style="height:1.0rem"></div>

The goal is not to **describe AI**.  
The goal is to **understand how it works**.  
Do not remain at the surface of outputs.  
Demonstrate that you can think in **mechanisms, modules, and pipelines**.  
Don't just use AI.  
**Explain it.**

<div style="height:1.5rem"></div>

<hr style="height: 3px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>
