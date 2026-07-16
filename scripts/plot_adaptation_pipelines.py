import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def setup_figure(title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    return fig, ax

def draw_box(ax, x, y, w, h, text, color="#f0f0f0", font_size=12, text_color="black"):
    rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=font_size, color=text_color, wrap=True)
    return x + w/2, y, x + w/2, y + h

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
    if text:
        ax.text((x1+x2)/2 + 2, (y1+y2)/2, text, ha='left', va='center', fontsize=10, style='italic')

def draw_epoch_array(ax, x, y, w, h, colors, labels=None):
    n = len(colors)
    w_ep = w / n
    for i in range(n):
        rect = patches.Rectangle((x + i*w_ep, y), w_ep, h, facecolor=colors[i], edgecolor='white', lw=1)
        ax.add_patch(rect)
        if labels and labels[i]:
            ax.text(x + i*w_ep + w_ep/2, y + h/2, labels[i], ha='center', va='center', fontsize=8, color='white', fontweight='bold')

def plot_fraction_pipeline(out_path):
    fig, ax = setup_figure("Adaptation Fraction Sweep (The Upper Bound / Random CV)")
    
    draw_box(ax, 20, 85, 60, 8, "Iterate through Subjects (1 to 21)\nExtract Thresholded 'Good' Epochs", "#e8f4f8")
    draw_arrow(ax, 50, 85, 50, 78)
    
    draw_box(ax, 20, 70, 60, 8, "Randomly Shuffle all Epochs for Subject", "#ffeaa7")
    draw_arrow(ax, 50, 70, 50, 63)
    
    draw_box(ax, 10, 50, 80, 13, "For each target fraction (e.g. 30%):\nPerform 10-Fold Cross Validation over shuffled array", "#f0f0f0")
    
    colors_run1 = ['#ff7675']*3 + ['#74b9ff']*7
    draw_epoch_array(ax, 15, 56, 70, 4, colors_run1, labels=["Adapt"]*3 + ["Evaluate"]*7)
    ax.text(12, 58, "Run 1", ha='right', va='center', fontweight='bold')
    
    colors_run2 = ['#74b9ff']*1 + ['#ff7675']*3 + ['#74b9ff']*6
    draw_epoch_array(ax, 15, 51, 70, 4, colors_run2, labels=["Eval"]*1 + ["Adapt"]*3 + ["Evaluate"]*6)
    ax.text(12, 53, "Run 2", ha='right', va='center', fontweight='bold')
    
    ax.text(50, 48, "... Repeats 10 times sliding the Adapt block ...", ha='center', va='center', style='italic')
    
    draw_arrow(ax, 50, 46, 50, 39)
    draw_box(ax, 25, 31, 50, 8, "Average the 10 Cross-Validation runs", "#e8f4f8")
    
    draw_arrow(ax, 50, 31, 50, 24)
    draw_box(ax, 15, 12, 70, 12, "Result: Because epochs were shuffled, 'Adapt' epochs\nwere randomly scattered across the entire 2 hours.\nThis leaks future data into the adaptation set!", "#ff9ff3")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_fixed_window_pipeline(out_path):
    fig, ax = setup_figure("Fixed Window Sweep (The Real-Time Continuous Simulation)")
    
    draw_box(ax, 20, 85, 60, 8, "Iterate through Subjects (1 to 21)\nExtract Thresholded 'Good' Epochs", "#e8f4f8")
    draw_arrow(ax, 50, 85, 50, 78)
    
    draw_box(ax, 20, 70, 60, 8, "Keep Epochs in Exact Chronological Order\n(DO NOT SHUFFLE)", "#55efc4")
    draw_arrow(ax, 50, 70, 50, 63)
    
    draw_box(ax, 10, 48, 80, 15, "For each target window size (e.g. 5 minutes = 37 contiguous epochs):\nPick 10 Random Starting Indexes to find the Average", "#f0f0f0")
    
    colors_run1 = ['#74b9ff']*2 + ['#ff7675']*3 + ['#74b9ff']*5
    draw_epoch_array(ax, 15, 56, 70, 4, colors_run1, labels=["Eval"]*2 + ["Adapt"]*3 + ["Evaluate"]*5)
    ax.text(12, 58, "Run 1", ha='right', va='center', fontweight='bold')
    
    colors_run2 = ['#74b9ff']*6 + ['#ff7675']*3 + ['#74b9ff']*1
    draw_epoch_array(ax, 15, 51, 70, 4, colors_run2, labels=["Evaluate"]*6 + ["Adapt"]*3 + ["Ev"]*1)
    ax.text(12, 53, "Run 2", ha='right', va='center', fontweight='bold')
    
    ax.text(50, 46, "... Repeats 10 times selecting random contiguous blocks ...", ha='center', va='center', style='italic')
    
    draw_arrow(ax, 50, 44, 50, 37)
    draw_box(ax, 25, 29, 50, 8, "Average the 10 contiguous runs", "#e8f4f8")
    
    draw_arrow(ax, 50, 29, 50, 22)
    draw_box(ax, 15, 10, 70, 12, "Result: Effectively simulates a continuous BCI sliding window.\nGuarantees N minutes of unbroken recent physiological data\nwith NO future leakage.", "#a29bfe")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_chronological_pipeline(out_path):
    fig, ax = setup_figure("Chronological Sweep (The Calibration Phase Simulation)")
    
    draw_box(ax, 20, 85, 60, 8, "Iterate through Subjects (1 to 21)\nExtract Thresholded 'Good' Epochs", "#e8f4f8")
    draw_arrow(ax, 50, 85, 50, 78)
    
    draw_box(ax, 20, 70, 60, 8, "Keep Epochs in Exact Chronological Order\n(DO NOT SHUFFLE)", "#55efc4")
    draw_arrow(ax, 50, 70, 50, 63)
    
    draw_box(ax, 10, 44, 80, 19, "For each target clock time (e.g. 10 minutes):\nStrictly use epochs before 10:00 to predict epochs after 10:00", "#f0f0f0")
    
    ax.plot([15, 85], [56, 56], color='black', lw=2)
    ax.plot([30, 30], [53, 59], color='red', lw=3, ls='--')
    ax.text(30, 61, "Threshold: 10:00", color='red', ha='center', fontweight='bold')
    
    colors_run1 = ['#ff7675']*2 + ['#74b9ff']*8
    draw_epoch_array(ax, 15, 54, 70, 4, colors_run1, labels=["Adapt"]*2 + ["Evaluate"]*8)
    ax.text(12, 56, "Single\nRun", ha='right', va='center', fontweight='bold')
    
    ax.text(50, 49, "Notice: 10 mins of clock time might only yield 2 'good' epochs\nif the subject was mostly ambiguous!", ha='center', va='center', style='italic', color='darkred')
    
    draw_arrow(ax, 50, 44, 50, 37)
    draw_box(ax, 25, 29, 50, 8, "NO Cross-Validation! (Done 1x per subject)", "#fab1a0")
    
    draw_arrow(ax, 50, 29, 50, 22)
    draw_box(ax, 15, 10, 70, 12, "Result: Simulates calibrating only once at the beginning of the drive.\nSuffers from 'drift' because the initial calibration\nbecomes irrelevant 2 hours later.", "#ffeaa7")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs("data/results/Final_Plots_All/Pipelines", exist_ok=True)
    plot_fraction_pipeline("data/results/Final_Plots_All/Pipelines/pipeline_fraction_sweep.png")
    plot_fixed_window_pipeline("data/results/Final_Plots_All/Pipelines/pipeline_fixed_window.png")
    plot_chronological_pipeline("data/results/Final_Plots_All/Pipelines/pipeline_chronological.png")
    print("Successfully generated all pipeline diagrams.")
