!pip install -q networkx scikit-learn matplotlib pandas tqdm lxml beautifulsoup4 plotly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import networkx as nx
import plotly.express as px
import glob
from sklearn.manifold import TSNE
from collections import defaultdict
from tqdm import tqdm
from bs4 import BeautifulSoup
from io import StringIO
SIMLEX_PATH = '/content/SimLex-999.txt'
HTML_PATH_PATTERN = '/content/Cue_Target_Pairs*.html'

plt.figure(figsize=(8, 6))
palette = sns.color_palette("gray", n_colors=4)
colors = df['concQ'].map({1: palette[0], 2: palette[1], 3: palette[2], 4: palette[3]})
plt.scatter(df['Assoc(USF)'], df['SimLex999'], c=colors, s=50, edgecolor='black')
plt.xlabel('Associative Strength (USF)')
plt.ylabel('Semantic Similarity (SimLex999)')
plt.title('Semantic vs Associative Strength Colored by Concreteness (concQ)')
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 10)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=palette[0], edgecolor='black', label='concQ = 1 (abstract)'),
    Patch(facecolor=palette[1], edgecolor='black', label='concQ = 2'),
    Patch(facecolor=palette[2], edgecolor='black', label='concQ = 3'),
    Patch(facecolor=palette[3], edgecolor='black', label='concQ = 4 (concrete)'),
]
plt.legend(handles=legend_elements, title='Concreteness Level')
plt.tight_layout()
plt.show()

simlex = pd.read_csv(SIMLEX_PATH, sep='\t')
words = set(simlex['word1']) | set(simlex['word2'])
concreteness = {}
for _, row in simlex.iterrows():
    concreteness[row['word1']] = row['conc(w1)']
    concreteness[row['word2']] = row['conc(w2)']

html_files = sorted(glob.glob(HTML_PATH_PATTERN))
all_pairs = []

for file in html_files:
      with open(file, 'r', encoding='ISO-8859-1') as f:
          soup = BeautifulSoup(f, 'lxml')
          pre = soup.find('pre')
          if pre:
              csv_text = pre.get_text()
              df = pd.read_csv(StringIO(csv_text))
              df.columns = [c.lower().strip() for c in df.columns]
              if {'cue', 'target', 'normed?', 'fsg'}.issubset(df.columns):
                  filtered = df[df['normed?'].str.upper() == 'YES']
                  all_pairs.append(filtered[['cue', 'target', 'fsg']])
usf = pd.concat(all_pairs, ignore_index=True)

G = nx.DiGraph()
for _, row in usf.iterrows():
    cue = str(row['cue']).lower()
    target = str(row['target']).lower()
    fsg = row.get('fsg')
    if pd.notna(fsg) and fsg > 0:
        G.add_edge(cue, target, weight=1 / fsg)
shortest_paths = defaultdict(dict)
for source in tqdm(words):
    for target in words:
        if source == target:
            shortest_paths[source][target] = 0
        elif source not in G or target not in G:
            shortest_paths[source][target] = 0
        else:
            try:
                shortest_paths[source][target] = nx.shortest_path_length(G, source=source, target=target, weight='weight')
            except nx.NetworkXNoPath:
                shortest_paths[source][target] = 0

vocab = sorted(words)
dist_matrix = np.array([[shortest_paths[w1][w2] for w2 in vocab] for w1 in vocab])
coords = TSNE(
    n_components=2,
    metric='precomputed',
    init='random',
    perplexity=50,
    learning_rate=500,
    random_state=42
).fit_transform(dist_matrix)

x = coords[:, 0]
y = coords[:, 1]
labels = vocab
colors = [concreteness.get(word, 3.0) for word in vocab]

fig = px.scatter(
    x=x,
    y=y,
    hover_name=labels,
    color=colors,
    color_continuous_scale='Viridis',
    labels={'color': 'Concreteness'},
    title='Semantic Distance Map via USF Paths (Color = Concreteness)'
)

fig.update_traces(marker=dict(size=6), text=None)
fig.update_layout(height=800, width=1000, showlegend=False, dragmode='zoom', hovermode='closest')
fig.show()

