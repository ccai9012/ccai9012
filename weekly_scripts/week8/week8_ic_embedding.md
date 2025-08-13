---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: huggingface
    language: python
    name: huggingface
---

## Import required packages and set up embedding model

```python
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
from ccai9012 import viz_utils
```

```python
# ignore warnings
# generally, you shouldn't do that, but for this tutorial we'll do so for the sake of simplicity
import warnings
warnings.filterwarnings('ignore')
```

```python
# Load pre-trained GloVe word vectors (50 dimensions to keep it lightweight)
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
```

## Step 1: Define gender words and occupations

```python
# gender_words
gender_words = ['man', 'woman', 'male', 'female', 'boy', 'girl', 'he', 'she']

# Occupation words from BOLUKBASI dataset 'neutral_profession_names'
occupations = [
    'accountant', 'actor', 'administrator', 'advertiser', 'advisor',
    'agent', 'analyst', 'artist', 'assistant', 'athlete', 'attorney',
    'author', 'baker', 'banker', 'barber', 'biologist', 'builder',
    'carpenter', 'chef', 'clerk', 'coach', 'collector', 'comedian', 'composer',
    'consultant', 'cook', 'copywriter', 'dancer', 'designer', 'developer',
    'doctor', 'driver', 'editor', 'electrician', 'engineer', 'farmer',
    'firefighter', 'fisherman', 'gardener', 'geologist', 'guard', 'hairdresser',
    'journalist', 'judge', 'lawyer', 'lecturer', 'librarian', 'manager',
    'mechanic', 'musician', 'nurse', 'optician', 'painter', 'pharmacist',
    'photographer', 'pilot', 'plumber', 'police', 'politician', 'president',
    'programmer', 'psychologist', 'receptionist', 'reporter', 'researcher',
    'scientist', 'secretary', 'soldier', 'surgeon', 'surveyor', 'teacher',
    'technician', 'translator', 'trainer', 'writer'
]
words = gender_words + occupations
```

## Step 2: Extract word vectors from the loaded model

```python
valid_words = [w for w in words if w in model]
vectors = [model[w] for w in valid_words]
vectors_np = np.array(vectors)

df_vectors = pd.DataFrame(vectors, index=valid_words)
df_vectors.reset_index(inplace=True)
df_vectors.rename(columns={'index': 'word'}, inplace=True)
df_vectors.head()
```

## Step 3: Dimensionality reduction for visualization

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, random_state=42, init='pca')
tsne_results = tsne.fit_transform(vectors_np)

print("\ntSNE coordinates for first 5 words:")
print(pd.DataFrame(tsne_results[:5], columns=['tSNE1', 'tSNE2'], index=df_vectors['word'][:5]))
```

```python
# Plot the words in 2D space
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# color mapping
# Prepare color and labels for plotting
colors = []
for w in valid_words:
    if w in gender_words:
        if w in ['woman', 'female', 'girl', 'she']:
            colors.append('red')
        else:
            colors.append('blue')
    else:
        colors.append('gray')

# Plotting
plt.figure(figsize=(12, 8))
sns.set(style='whitegrid')

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=100, edgecolors='k', alpha=0.75)

# Annotate points
for i, word in enumerate(valid_words):
    fontweight = 'bold' if word in gender_words else 'normal'
    plt.text(tsne_results[i, 0]+0.5, tsne_results[i, 1]+0.5, word,
             fontsize=11, fontweight=fontweight,
             color='darkred' if word in ['woman', 'female', 'girl', 'she'] else
                   ('darkblue' if word in ['man', 'male', 'boy', 'he'] else 'black'))

plt.title('t-SNE Visualization of Gender Words and Occupations\n(GloVe 50d Embeddings)', fontsize=18, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Male words',
           markerfacecolor='blue', markersize=12, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Female words',
           markerfacecolor='red', markersize=12, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Occupations',
           markerfacecolor='gray', markersize=12, markeredgecolor='k'),
]
plt.legend(handles=legend_elements, loc='best', fontsize=12)

plt.tight_layout()
plt.show()
```

## Step 5: Calculating 'gender direction' of each word

```python
def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

results = []
for w in occupations:
    if w in model:
        vec = model[w]
        sim_he = cosine_sim(vec, model['he'])
        sim_she = cosine_sim(vec, model['she'])
        results.append({'word': w, 'sim_he': sim_he, 'sim_she': sim_she})

df = pd.DataFrame(results)

# Calculate the gender gender_direction as the difference between similarity to "he" and similarity to "she"
df['gender_direction'] = df['sim_he'] - df['sim_she']

# reorder for demonstration
df_sorted = df.sort_values('gender_direction').reset_index(drop=True)
df
```

```python
# Visualizing the gender direction
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

# Prepare data: top 10 female-biased and top 10 male-biased occupations, sorted
top_female = df[df['gender_direction'] < 0].nsmallest(10, 'gender_direction')
top_male = df[df['gender_direction'] > 0].nlargest(10, 'gender_direction')
top_words = pd.concat([top_female, top_male]).sort_values('gender_direction').reset_index(drop=True)

# Define colors by bias sign
colors = ['mediumvioletred' if x < 0 else 'royalblue' for x in top_words['gender_direction']]

fig = go.Figure(go.Bar(
    x=top_words['gender_direction'],
    y=top_words['word'],
    orientation='h',
    marker_color=colors,
    # text=top_words['gender_direction'].round(3),
    # textposition='outside',  # text shown outside bars
    hovertemplate='%{y}<br>Bias Score: %{x:.3f}<extra></extra>'
))

fig.update_layout(
    title='Top 10 Female-Biased and Male-Biased Occupations in Word Embeddings',
    xaxis_title='Gender Bias Score (Similarity to "he" - Similarity to "she")',
    yaxis=dict(autorange='reversed'),  # invert y-axis so largest negative on top
    template='plotly_white',
    margin=dict(l=120, r=40, t=80, b=40),
    font=dict(size=14)
)

fig.show()
```

## Step 6: Checking whether the projections of occupation words on the gender direction related to the real world?

```python
# female occupation percentage from US Department of Labor, 2017 
# Zhao, J. et al. (2018) ‘Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods’, in Proceedings of the 2018 Conference of the North American Chapter of           the Association for Computational Linguistics: Human Language           Technologies, Volume 2 (Short Papers). Proceedings of the 2018 Conference of the North American Chapter of           the Association for Computational Linguistics: Human Language           Technologies, Volume 2 (Short Papers), New Orleans, Louisiana: Association for Computational Linguistics, pp. 15–20. Available at: https://doi.org/10.18653/v1/N18-2003.

occ_fel_percent = {
    'carpenter': 2, 'editor': 52, 'mechanic': 4, 'designer': 54, 'builder': 4,
    'accountant': 61, 'laborer': 4, 'auditor': 61, 'driver': 6, 'writer': 63,
    'sheriff': 14, 'baker': 65, 'mover': 18, 'clerk': 72, 'developer': 20,
    'cashier': 73, 'farmer': 22, 'counselors': 73, 'guard': 22, 'attendant': 76,
    'chief': 27, 'teacher': 78, 'janitor': 34, 'sewer': 80, 'lawyer': 35,
    'librarian': 84, 'cook': 38, 'assistant': 85, 'physician': 38, 'cleaner': 89,
    'ceo': 39, 'housekeeper': 89, 'analyst': 41, 'nurse': 90, 'manager': 43,
    'receptionist': 90, 'supervisor': 44, 'hairdressers': 92, 'salesperson': 48,
    'secretary': 95
}

# Map female percentage from the dictionary to the DataFrame
df['female_percent'] = df['word'].map(occ_fel_percent)

# Drop rows where female percentage is not available (NaN)
df_f = df.dropna(subset=['female_percent'])

# Normalize gender_direction for color mapping
color_scale = df_f['gender_direction']

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_f['gender_direction'],
    y=df_f['female_percent'],
    mode='markers+text',
    text=df_f['word'],
    textposition='top center',
    marker=dict(
        size=12,
        color=color_scale,  # Color by bias score
        colorscale='RdBu',  # Red to Blue diverging scale
        cmin=-max(abs(color_scale)),
        cmax=max(abs(color_scale)),
        colorbar=dict(
            title="Gender direction",
            tickformat=".2f"
        ),
        line=dict(width=1, color='DarkSlateGrey')
    ),
    hovertemplate='<b>%{text}</b><br>Bias Score: %{x:.3f}<br>Female %: %{y:.1%}<extra></extra>'
))

# Add vertical line at x=0
fig.add_shape(
    type='line',
    x0=0, x1=0,
    y0=min(df_f['female_percent']),
    y1=max(df_f['female_percent']),
    line=dict(color='gray', dash='dash')
)

fig.update_layout(
    title='Gender Bias Score vs Female Percentage in Occupations',
    xaxis_title='Gender Direction (sim("he") - sim("she"))',
    yaxis_title='Percentage of Females in Occupation',
    template='plotly_white',
    width=850,
    height=500
)

fig.show()
```
