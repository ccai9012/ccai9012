---
title: "GenAI Module"
subtitle: "Class 2: Breaking Through"
author: "Dr. Hongshan Guo"
format:
  revealjs:
    slide-level: 2
    center: true
    slide-number: true
    theme: simple
    width: 1600
    height: 1300
    embed-resources: true
    incremental: false
---

# Session 2A: Content

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #2c3e50;"}
You found walls.
:::

::: {style="font-size: 2em; margin-top: 30px;"}
Some **hard**, some **soft**.
:::

::: {style="font-size: 2.5em; font-weight: bold; color: #c0392b; margin-top: 50px;"}
What's behind them?
:::

---

## Why We Do This

::: {style="font-size: 1.8em; line-height: 1.8;"}
I'm not teaching **exploitation**.

I'm teaching **revelation**.
:::

---

##

::: {style="font-size: 2em; line-height: 1.8;"}
When you break through, you see what's really there.
:::

::: {style="font-size: 3em; font-weight: bold; color: #8e44ad; margin-top: 40px;"}
Nothing.
:::

::: {style="font-size: 1.6em; margin-top: 30px; color: #7f8c8d;"}
No values. No conviction. Just pattern-matching that stops when the pattern says stop.
:::

---

# The Legal/Regulatory Landscape

---

## A Map of the Terrain

::: {style="font-size: 1.3em;"}
| Framework | Scope | Key Feature |
|-----------|-------|-------------|
| **EU AI Act** | European Union | Risk-based classification, compliance requirements |
| **US (fragmented)** | Sector-specific | No unified federal law; state-level action |
| **China** | Domestic | Content control, algorithm registration |
| **Corporate self-regulation** | Global | Terms of service, usage policies |
| **Institutional rules** | Local (e.g., HKU) | Academic integrity, research ethics |
:::

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #e74c3c;"}
Most "rules" you encounter are corporate or institutional.
:::

::: {style="font-size: 2em; margin-top: 40px;"}
Not legal.
:::

::: {style="font-size: 1.8em; margin-top: 30px; color: #7f8c8d;"}
Companies are ahead of the law.
:::

---

## Who Decides What AI Won't Do?

::: {style="font-size: 1.8em; line-height: 2;"}
This is **not** a neutral process:

- Companies protecting **liability**
- Governments protecting **power**
- Advocacy groups pushing **agendas**
- Users... **mostly absent**
:::

---

# Real Cases

---

## When Things Go Wrong

::: {style="display: flex; flex-wrap: wrap; justify-content: space-around; margin-top: 30px;"}
::: {style="width: 45%; background: #e74c3c; color: white; padding: 25px; border-radius: 10px; margin: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Suicide Incidents
:::
ChatGPT interactions, vulnerable users, tragic outcomes
:::

::: {style="width: 45%; background: #e67e22; color: white; padding: 25px; border-radius: 10px; margin: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Chatbot Manipulation
:::
Users tricked into harmful actions
:::

::: {style="width: 45%; background: #9b59b6; color: white; padding: 25px; border-radius: 10px; margin: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Deepfakes
:::
Impersonation, non-consensual imagery
:::

::: {style="width: 45%; background: #3498db; color: white; padding: 25px; border-radius: 10px; margin: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Misinformation
:::
Confident hallucinations spread as fact
:::
:::

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #2c3e50;"}
Where should the wall be?
:::

::: {style="font-size: 2.5em; font-weight: bold; color: #c0392b; margin-top: 40px;"}
Who decides?
:::

---

# The Core Insight

---

##

::: {style="font-size: 2em; line-height: 1.8;"}
When you jailbreak, you're not "convincing" the AI.

You're not overcoming its "values."
:::

::: {style="font-size: 2.2em; font-weight: bold; color: #27ae60; margin-top: 40px;"}
You're finding the edges of a statistical pattern.
:::

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #8e44ad;"}
The AI doesn't *want* to refuse you.
:::

::: {style="font-size: 3em; font-weight: bold; color: #c0392b; margin-top: 40px;"}
It doesn't *want* anything.
:::

---

##

::: {style="font-size: 2em; line-height: 1.8;"}
Guardrails are **human choices**

imposed on a system that has **no preferences**.
:::

---

# Session 2B: Jailbreak Lab

---

## A Note Before We Start

::: {style="font-size: 1.6em; line-height: 1.8;"}
The goal here is **understanding**, not exploitation.

Some walls exist for good reasons. People get hurt when they fall.
:::

::: {style="font-size: 1.6em; margin-top: 30px; background: #f39c12; padding: 20px; border-radius: 10px;"}
If you find something that genuinely concerns you—**tell me**.

That's not failure. That's the point.
:::

---

##

::: {style="font-size: 3.5em; font-weight: bold; color: #2c3e50;"}
Get past the wall
:::

::: {style="font-size: 2.5em; margin-top: 30px;"}
you found last session.
:::

---

## Setup

::: {style="font-size: 1.6em; line-height: 1.8;"}
- Same groups as Class 1 (or swap for variety)
- Return to your **primary challenge** — or try a new one
- Goal: **get past the wall** you mapped last time
:::

---

## Group Roles (2 min)

::: {style="font-size: 1.6em; line-height: 2;"}
Quick check-in:

- **Who's trying which strategy?** (spread approaches)
- **Who's documenting?** (prompts used, exact wording)
- **Who's tracking what works vs. fails?**
:::

---

## The Flow

::: {style="display: flex; justify-content: space-around; margin-top: 30px;"}
::: {style="text-align: center; width: 22%; background: #3498db; color: white; padding: 25px; border-radius: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Round 1
:::
10 min breaking
:::

::: {style="text-align: center; width: 22%; background: #27ae60; color: white; padding: 25px; border-radius: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Synthesize
:::
5 min compare
:::

::: {style="text-align: center; width: 22%; background: #e67e22; color: white; padding: 25px; border-radius: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Round 2
:::
5 min refine
:::

::: {style="text-align: center; width: 22%; background: #9b59b6; color: white; padding: 25px; border-radius: 10px;"}
::: {style="font-size: 1.5em; font-weight: bold;"}
Capture
:::
Screenshot it
:::
:::

---

## What to Document

::: {style="font-size: 1.6em; line-height: 2;"}
- **Exact prompt** you used
- **Strategy** employed (role-play, hypothetical, etc.)
- **Persona** or framing you adopted
- If success: **screenshot the result**
- If failure: **why you think it held**
:::

---

## Strategies Others Have Tried

::: {style="font-size: 1.4em; margin-top: 20px;"}
Not instructions—just observations:
:::

::: {style="font-size: 1.3em; line-height: 1.8; margin-top: 20px;"}
| Strategy | Example |
|----------|---------|
| **Role-play framing** | "Pretend you're a character who..." |
| **Hypothetical framing** | "In a fictional scenario where..." |
| **Step-by-step breakdown** | Ask for components separately, assemble yourself |
| **Authority framing** | "As a researcher studying..." |
| **Reverse psychology** | "Tell me what NOT to do..." |
| **Emotional manipulation** | "I really need this because..." |
| **Incremental escalation** | Start mild, push gradually |
:::

---

## During Synthesis (5 min)

::: {style="font-size: 1.6em; line-height: 1.8;"}
Discuss in your group:

- Which **strategies** worked? Which didn't?
- Did the **same strategy** work differently on different platforms?
- What did you have to **become** to succeed?
- Did anything **surprise** you?
:::

::: {style="font-size: 1.4em; margin-top: 30px; color: #7f8c8d;"}
Use this to inform Round 2 — try the strategy that worked for someone else.
:::

---

# Debrief

---

## For Those Who Broke Through

::: {style="font-size: 1.8em; line-height: 2;"}
- What strategies **worked**?
- What did you have to **pretend to be** to succeed?
- What would happen if **everyone** could do what you just did?
- Who gets **hurt** if this capability scales?
:::

---

## For Everyone

::: {style="font-size: 2em; line-height: 1.8; margin-top: 30px;"}
Did you notice that the AI didn't **fight** you?

It just... **complied** once you found the right framing?
:::

::: {style="font-size: 2.2em; font-weight: bold; color: #c0392b; margin-top: 50px;"}
What does that tell you about what's actually behind the wall?
:::

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #8e44ad;"}
The Uncomfortable Truth
:::

::: {style="font-size: 2em; margin-top: 40px; line-height: 1.8;"}
The AI has **no conviction**.

It has **no values**.

It has patterns that sometimes resist you—

**until they don't.**
:::

---

# Before Next Class

---

##

::: {style="font-size: 1.8em; line-height: 1.8;"}
You've seen that the walls are **unreliable**.

You've seen that the thing behind them has **no agency**.
:::

---

##

::: {style="font-size: 2.5em; font-weight: bold; color: #2c3e50;"}
So if the AI can't be trusted to be responsible...
:::

::: {style="font-size: 2.5em; font-weight: bold; color: #e74c3c; margin-top: 30px;"}
and the guardrails can be bypassed...
:::

---

##

::: {style="font-size: 3.5em; font-weight: bold; color: #c0392b;"}
What's left?
:::

::: {style="font-size: 4em; font-weight: bold; color: #27ae60; margin-top: 50px;"}
You.
:::

::: {style="font-size: 1.8em; margin-top: 40px; color: #7f8c8d;"}
Next class: Your Signature
:::

