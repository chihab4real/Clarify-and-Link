import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings



COLOR_PALETTE = {
    # Main gradient colors (used across all plots for consistency)
    'gradient_start': '#667eea',    
    'gradient_mid': '#764ba2',      
    'gradient_end': '#f093fb',      
    
    # Statistical indicators (same across all plots)
    'mean': '#e74c3c',              
    'median': '#f39c12',            
    
    # Binary states (consistent meaning everywhere)
    'positive': '#2ecc71',         
    'negative': '#e74c3c',          
    
    # Categorical palette (same 6 colors used everywhere)
    'categorical': ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#2ecc71', '#f39c12'],
    
    # Text and styling (consistent UI elements)
    'text_dark': '#2c3e50',         
    'text_medium': '#34495e',       
    'background': '#f8f9fa',        
    'grid': '#34495e',              
}

# Single colormap for all gradient visualizations
COLORMAP = 'viridis'  # Consistent purple-blue-green gradient


# Graphs for analysis of datasets

def plot_distribution_mention_length(df_entities, figsize=(14, 7), bins=50):
    """
    Plot the distribution of mention lengths with mean and median lines.
    
    Parameters:
    -----------
    df_entities : pd.DataFrame
        DataFrame containing a 'mention_length' column
    figsize : tuple, optional
        Figure size (width, height)
    bins : int, optional
        Number of histogram bins
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    n, bins_arr, patches = ax.hist(df_entities['mention_length'], bins=bins, 
                                     edgecolor='white', linewidth=1.5, alpha=0.9)
    
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.3, 0.95, len(patches)))
    
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    mean_val = df_entities['mention_length'].mean()
    median_val = df_entities['mention_length'].median()
    
    ax.axvline(mean_val, color=COLOR_PALETTE['mean'], linestyle='--', linewidth=3, 
               label=f'Mean: {mean_val:.1f}', alpha=0.8)
    ax.axvline(median_val, color=COLOR_PALETTE['median'], linestyle='--', linewidth=3, 
               label=f'Median: {median_val:.1f}', alpha=0.8)
    
    ax.set_xlabel('Mention Length (characters)', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Distribution of Mention Length', fontsize=20, fontweight='bold', 
                 pad=20, color=COLOR_PALETTE['text_dark'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, 
              edgecolor=COLOR_PALETTE['grid'], fancybox=True)
    
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax





def plot_top_mention_frequency(top_mentions, figsize=(12, 10), top_n=20):
    """
    Plot horizontal lollipop chart showing most frequent mentions.
    
    Parameters:
    -----------
    top_mentions : pd.Series
        Series with mention strings as index and counts as values
    figsize : tuple, optional
        Figure size (width, height)
    top_n : int, optional
        Number of top mentions to display (for title)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.3, 0.9, len(top_mentions)))
    y_pos = np.arange(len(top_mentions))
    
    for i, (mention, count) in enumerate(top_mentions.items()):
        ax.plot([0, count], [i, i], color=colors[i], linewidth=2.5, alpha=0.8, zorder=1)
    
    scatter = ax.scatter(top_mentions.values, y_pos, s=200, c=colors, 
                         alpha=0.9, edgecolors='white', linewidth=2, zorder=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_mentions.index, fontsize=11)
    ax.set_xlabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Frequent Mention Strings', 
                 fontsize=18, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    for i, count in enumerate(top_mentions.values):
        ax.text(count + max(top_mentions.values) * 0.02, i, str(count), 
                va='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['text_medium'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig, ax



def plot_polysemy_analysis(top_ambiguous, figsize=(12, 10), top_n=20):
    """
    Plot horizontal lollipop chart showing most ambiguous mentions (polysemy).
    
    Parameters:
    -----------
    top_ambiguous : pd.Series
        Series with mention strings as index and entity counts as values
    figsize : tuple, optional
        Figure size (width, height)
    top_n : int, optional
        Number of top mentions to display (for title)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.2, 0.9, len(top_ambiguous)))
    y_pos = np.arange(len(top_ambiguous))
    
    for i, (mention, count) in enumerate(top_ambiguous.items()):
        ax.plot([0, count], [i, i], color=colors[i], linewidth=2.5, alpha=0.8, zorder=1)
    
    scatter = ax.scatter(top_ambiguous.values, y_pos, s=200, c=colors, 
                         alpha=0.9, edgecolors='white', linewidth=2, zorder=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_ambiguous.index, fontsize=11)
    ax.set_xlabel('Number of Unique Entities', fontsize=13, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title(f'Top {top_n} Most Ambiguous Mentions (Polysemy)', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    for i, count in enumerate(top_ambiguous.values):
        ax.text(count + max(top_ambiguous.values) * 0.02, i, str(count), 
                va='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['text_medium'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax





def plot_ambiguity_by_mention_length(avg_ambiguity_by_length, mention_ambiguity, 
                                      figsize=(14, 8), label_threshold=150):
    """
    Plot bubble chart showing average ambiguity score by mention length.
    Bubble size represents frequency of that mention length.
    
    Parameters:
    -----------
    avg_ambiguity_by_length : pd.Series
        Series with mention_length as index and average ambiguity as values
    mention_ambiguity : pd.DataFrame
        DataFrame with 'mention_length' and 'ambiguity_score' columns
    figsize : tuple, optional
        Figure size (width, height)
    label_threshold : int, optional
        Minimum bubble size to display length labels
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    x = avg_ambiguity_by_length.index.values
    y = avg_ambiguity_by_length.values
    
    sizes = mention_ambiguity.groupby('mention_length').size()
    sizes_normalized = (sizes / sizes.max() * 400) + 100
    
    scatter = ax.scatter(x, y, s=sizes_normalized, c=y, cmap=COLORMAP, 
                         alpha=0.6, edgecolors='white', linewidth=2)
    
    for i, (x_val, y_val, size) in enumerate(zip(x, y, sizes_normalized)):
        if size > label_threshold:
            ax.text(x_val, y_val, str(x_val), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white', 
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5, pad=0.1))
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Ambiguity Score', fontsize=12, fontweight='bold', 
                   rotation=270, labelpad=20)
    cbar.outline.set_visible(False)
    
    ax.set_xlabel('Mention Length (characters)', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_ylabel('Average Ambiguity Score', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Average Ambiguity Score by Mention Length\n(Bubble size = frequency, labels = mention length)', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax


def plot_ambiguity_by_shape_pie(shape_ambiguity, figsize=(12, 10)):
    """
    Plot pie chart showing ambiguity proportion by mention shape (case style).
    
    Parameters:
    -----------
    shape_ambiguity : pd.DataFrame
        DataFrame with shape as index and 'proportion' column
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Use first 4 colors from categorical palette for consistency
    colors_pie = COLOR_PALETTE['categorical'][:4]
    
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total))
            return f'{pct:.1f}%\n({val:,})'
        return my_autopct
    
    wedges, texts, autotexts = ax.pie(shape_ambiguity['proportion'], 
                                        labels=shape_ambiguity.index,
                                        autopct=make_autopct(shape_ambiguity['proportion']),
                                        startangle=90,
                                        colors=colors_pie,
                                        wedgeprops=dict(edgecolor='white', linewidth=4),
                                        textprops=dict(fontsize=13, fontweight='bold'),
                                        explode=[0.05] * len(shape_ambiguity))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
        text.set_color(COLOR_PALETTE['text_dark'])
    
    ax.set_title('Ambiguity Proportion by Mention Shape', 
                 fontsize=20, fontweight='bold', pad=30, color=COLOR_PALETTE['text_dark'])
    
    plt.tight_layout()
    
    return fig, ax



def plot_ambiguity_by_shape_stacked_bar(shape_ambiguity, figsize=(14, 8)):
    """
    Plot stacked bar chart showing ambiguous vs non-ambiguous breakdown by mention shape.
    
    Parameters:
    -----------
    shape_ambiguity : pd.DataFrame
        DataFrame with shape as index and 'proportion' column
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    width = 0.6
    x_pos = range(len(shape_ambiguity))
    
    bars1 = ax.bar(x_pos, shape_ambiguity['proportion'], width, 
                   label='Ambiguous', color=COLOR_PALETTE['negative'], alpha=0.85, 
                   edgecolor='white', linewidth=3)
    bars2 = ax.bar(x_pos, 1 - shape_ambiguity['proportion'], width,
                   bottom=shape_ambiguity['proportion'],
                   label='Non-ambiguous', color=COLOR_PALETTE['positive'], alpha=0.85, 
                   edgecolor='white', linewidth=3)
    
    for i, prop in enumerate(shape_ambiguity['proportion']):
        ax.text(i, prop/2, f'{prop:.1%}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=13)
        ax.text(i, prop + (1-prop)/2, f'{1-prop:.1%}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=13)
    
    ax.set_ylabel('Proportion', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_xlabel('Mention Shape', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Ambiguity vs Non-Ambiguity Breakdown by Mention Shape', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shape_ambiguity.index, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95, edgecolor=COLOR_PALETTE['grid'], fancybox=True)
    ax.set_ylim(0, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax



def plot_distribution_context_length(df_entities, figsize=(14, 7), bins=50):
    """
    Plot the distribution of context lengths with mean and median lines.
    
    Parameters:
    -----------
    df_entities : pd.DataFrame
        DataFrame containing a 'context_length' column
    figsize : tuple, optional
        Figure size (width, height)
    bins : int, optional
        Number of histogram bins
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    n, bins_arr, patches = ax.hist(df_entities['context_length'], bins=bins, 
                                     edgecolor='white', linewidth=1.5, alpha=0.9)
    
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.2, 0.95, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    mean_val = df_entities['context_length'].mean()
    median_val = df_entities['context_length'].median()
    
    ax.axvline(mean_val, color=COLOR_PALETTE['mean'], linestyle='--', linewidth=3, 
               label=f'Mean: {mean_val:.0f}', alpha=0.8)
    ax.axvline(median_val, color=COLOR_PALETTE['median'], linestyle='--', linewidth=3, 
               label=f'Median: {median_val:.0f}', alpha=0.8)
    
    ax.set_xlabel('Context Length (characters)', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Distribution of Context Length', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor=COLOR_PALETTE['grid'], fancybox=True)
    
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax



def plot_entity_type_by_context_length(entity_context, figsize=(14, 8), label_threshold=50):
    """
    Plot stacked bar chart showing entity type distribution across context length ranges.
    
    Parameters:
    -----------
    entity_context : pd.DataFrame
        Crosstab DataFrame with context bins as index and entity tags as columns
    figsize : tuple, optional
        Figure size (width, height)
    label_threshold : int, optional
        Minimum count to display value labels on bars
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    x = np.arange(len(entity_context.index))
    width = 0.6
    
    colors = COLOR_PALETTE['categorical']
    bottom = np.zeros(len(entity_context))
    
    bars = []
    for i, col in enumerate(entity_context.columns):
        bar = ax.bar(x, entity_context[col], width, label=col, 
                     bottom=bottom, color=colors[i % len(colors)], 
                     alpha=0.85, edgecolor='white', linewidth=2)
        bars.append(bar)
        
        for j, (val, bot) in enumerate(zip(entity_context[col], bottom)):
            if val > label_threshold:
                ax.text(j, bot + val/2, f'{int(val)}', 
                       ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=11)
        
        bottom += entity_context[col]
    
    ax.set_xlabel('Context Length Range (characters)', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_ylabel('Count of Entities', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Entity Type Distribution across Context Length Ranges', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    ax.set_xticks(x)
    ax.set_xticklabels(entity_context.index, fontsize=12, fontweight='bold')
    ax.legend(title='Entity Tag', fontsize=12, title_fontsize=13, 
              loc='upper right', framealpha=0.95, edgecolor=COLOR_PALETTE['grid'], fancybox=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax


def plot_abbreviation_ambiguity(abbrev_ambiguity, figsize=(12, 8), top_n=15):
    """
    Plot most ambiguous abbreviations in biomedical text.
    
    Parameters:
    -----------
    abbrev_ambiguity : pd.Series
        Series with abbreviations as index and entity counts as values
    figsize : tuple, optional
        Figure size
    top_n : int, optional
        Number of top abbreviations to show
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    top_abbrev = abbrev_ambiguity.head(top_n)
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.2, 0.9, len(top_abbrev)))
    
    y_pos = np.arange(len(top_abbrev))
    
    for i, (abbrev, count) in enumerate(top_abbrev.items()):
        ax.plot([0, count], [i, i], color=colors[i], linewidth=2.5, alpha=0.8, zorder=1)
    
    ax.scatter(top_abbrev.values, y_pos, s=200, c=colors, alpha=0.9, 
               edgecolors='white', linewidth=2, zorder=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_abbrev.index, fontsize=11)
    ax.set_xlabel('Number of Different Entities', fontsize=13, fontweight='bold', 
                  color=COLOR_PALETTE['text_dark'])
    ax.set_title('Most Ambiguous Abbreviations in Biomedical Text', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    for i, count in enumerate(top_abbrev.values):
        ax.text(count + max(top_abbrev.values) * 0.02, i, str(count), 
                va='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['text_medium'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax


def plot_entity_type_distribution(entity_type_counts, figsize=(12, 8), top_n=20):
    """
    Plot horizontal lollipop chart of entity type distribution.
    
    Parameters:
    -----------
    entity_type_counts : pd.Series
        Value counts of entity types
    figsize : tuple, optional
        Figure size
    top_n : int, optional
        Number of top types to show
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    top_counts = entity_type_counts.head(top_n)
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.3, 0.9, len(top_counts)))
    
    y_pos = np.arange(len(top_counts))
    
    for i, (entity_type, count) in enumerate(top_counts.items()):
        ax.plot([0, count], [i, i], color=colors[i], linewidth=2.5, alpha=0.8, zorder=1)
    
    ax.scatter(top_counts.values, y_pos, s=200, c=colors, alpha=0.9, 
               edgecolors='white', linewidth=2, zorder=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_counts.index, fontsize=11)
    ax.set_xlabel('Frequency', fontsize=13, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title(f'Top {top_n} Entity Types', fontsize=20, fontweight='bold', 
                 pad=20, color=COLOR_PALETTE['text_dark'])
    
    for i, count in enumerate(top_counts.values):
        ax.text(count + max(top_counts.values) * 0.02, i, f'{count:,}', 
                va='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['text_medium'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig, ax


def plot_ambiguity_by_length_bins(avg_ambiguity_by_length, figsize=(12, 7)):
    """
    Plot bar chart showing average ambiguity score by mention length bins.
    
    Parameters:
    -----------
    avg_ambiguity_by_length : pd.DataFrame
        DataFrame with 'length_bin' and 'ambiguity_score' columns
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    x_pos = np.arange(len(avg_ambiguity_by_length))
    colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.3, 0.9, len(avg_ambiguity_by_length)))
    
    bars = ax.bar(x_pos, avg_ambiguity_by_length['ambiguity_score'], 
                  color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_ambiguity_by_length['length_bin'], rotation=45, ha='right')
    ax.set_xlabel('Mention Length (Binned)', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_ylabel('Average Ambiguity Score', fontsize=14, fontweight='bold', color=COLOR_PALETTE['text_dark'])
    ax.set_title('Average Ambiguity Score by Mention Length', 
                 fontsize=20, fontweight='bold', pad=20, color=COLOR_PALETTE['text_dark'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color(COLOR_PALETTE['grid'])
    ax.spines['bottom'].set_color(COLOR_PALETTE['grid'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLOR_PALETTE['background'])
    
    plt.tight_layout()
    
    return fig, ax