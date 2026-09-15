
# A0: Applied Exercise

# _How to Talk to an LLM_

<div style="height:1.5rem"></div>

Please read these notes **before beginning your AI experiments**.

The aim is to keep your conversations sufficiently **consistent, controlled, and traceable** that you can later compare what the model did across different questions, site collections, and stages of the exercise.

<div style="height:1.5rem"></div>

<hr style="height: 6px; border: none; background-color: #000;">

<div style="height:1.5rem"></div>

<a id="platform"></a>

## 1. Choose a Platform

Use a multimodal LLM that can accept both **images and text**.

Possible platforms include:

- **Gemini**.
- **HKU ChatGPT**.
- Another course-approved multimodal LLM, where appropriate.

For a comparable set of tests, try to use the **same platform / model consistently**.

If you deliberately compare different models, treat them as separate experiments and keep the visual evidence and question wording as similar as possible.

<div style="
    border-left: 5px solid #000;
    padding: 0.8rem 1rem;
    margin: 1.5rem 0;
    background: #f2f2f2;
">
<strong>IMPORTANT:</strong> Consistency matters more than choosing the supposedly “best” model. If you change models, prompting style, images, and questions at the same time, it becomes difficult to understand what caused the difference in the response.
</div>

<div style="height:1.5rem"></div>

<hr>

<div style="height:1.5rem"></div>

<a id="interrogate"></a>

## 2. During INTERROGATE

Treat each independent question as a **fresh test**.

For each test:

1. Start a **new conversation**.
2. Upload only the image(s) you intend to test.
3. Ask the question directly.
4. Avoid giving hints, explanations, or corrections beforehand.
5. Record the model's **first response**.

The first response is particularly important because it shows what the model inferred before you began guiding or correcting it.

<div style="
    border-left: 5px solid #000;
    padding: 0.8rem 1rem;
    margin: 1.5rem 0;
    background: #f2f2f2;
">
<strong>INTERROGATE = CONTROL FIRST, CONVERSATION SECOND.</strong><br>
Start from a clean state, record the first response, and only then begin probing the model further.
</div>

<div style="height:1.5rem"></div>

<hr>

<div style="height:1.5rem"></div>

## 3. Follow Up After the First Response

Once the first response has been recorded, continue the same conversation and investigate the answer.

Useful follow-up prompts include:

- What visual evidence supports your answer?
- Which part of the image are you referring to?
- How certain are you?
- What are you assuming?
- Could there be another explanation?
- What information is missing?
- Look again. Would you revise your answer?

You do not need to use all of these.

Use follow-up questions when they help reveal whether the model can:

- justify its answer;
- recognise uncertainty;
- correct itself;
- become more precise;
- maintain an unsupported claim;
- contradict an earlier response.

<div style="height:1.5rem"></div>

<hr>

<div style="height:1.5rem"></div>

<a id="comparison"></a>

## 4. Keep Comparisons Consistent

When asking the same **systematic question** across several site collections:

- Keep the wording as similar as possible.
- Avoid adding extra explanation for some collections but not others.
- Keep the amount and type of visual evidence comparable where possible.

If you deliberately change the visual evidence — for example by adding another viewpoint or scale — make that change explicit.

A useful comparison might be:

**System only → System + Detail → Context + System + Detail**

The purpose is to understand what changes when the **evidence changes**, rather than when the prompting procedure changes unintentionally.

<div style="height:1.5rem"></div>

<hr>

<div style="height:1.5rem"></div>

<a id="transform"></a>

## 5. During TRANSFORM

During TRANSFORM, you may work much more openly with the AI.

You may:

- Explain your transformation intention.
- Provide context and constraints.
- Introduce references.
- Correct misunderstandings.
- Refine your instructions.
- Build on earlier outputs.
- Ask for targeted revisions.
- Combine the LLM with image generation, image editing, coding, or other tools where useful.

You do **not** need to restart the conversation for every prompt.

Here, accumulated context can help you develop the transformation more effectively.

<div style="
    border-left: 5px solid #000;
    padding: 0.8rem 1rem;
    margin: 1.5rem 0;
    background: #f2f2f2;
">
<strong>TRANSFORM = ITERATE.</strong><br>
Context, feedback, correction, and accumulated conversation are now part of the process.
</div>

<div style="height:1.5rem"></div>

<hr>

<div style="height:1.5rem"></div>

## 6. Keep Enough of a Record

You do not need to preserve every minor exchange.

Keep enough information that you can later identify:

- Which **platform / model** you used.
- Which **image(s)** were provided.
- Which **question or instruction** was given.
- The model's **first response** for important INTERROGATE tests.
- Important follow-up exchanges.
- Important prompts, corrections, or intermediate outputs during TRANSFORM.

For the exact material that needs to be submitted, see [**[Artefacts Checklist — INTERROGATE]**](a0_artefacts.html#interrogate) and [**[Artefacts Checklist — TRANSFORM]**](a0_artefacts.html#transform).

