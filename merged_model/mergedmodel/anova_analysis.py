#we have to reshape the main data into long-format data where each row is a displacement value with its associated treatment label.import pandas as pd
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
from pathlib import Path
import numpy as np


# 1. Load the dataset
data_file = "combined_mean_displacements.tsv"
df = pd.read_csv(data_file, sep="\t")

# 2. Reshape data from wide to long format
df_long = pd.melt(df, id_vars=["RunNumber"], value_vars=["Linear", "Radial", "Uniform"],
                  var_name="Treatment", value_name="Displacement")

# 3. Perform one-way ANOVA
linear = df_long[df_long["Treatment"] == "Linear"]["Displacement"]
radial = df_long[df_long["Treatment"] == "Radial"]["Displacement"]
uniform = df_long[df_long["Treatment"] == "Uniform"]["Displacement"]

anova_result = f_oneway(linear, radial, uniform)
f_stat = anova_result.statistic
p_val = anova_result.pvalue

# Save ANOVA results to a TSV file
anova_df = pd.DataFrame({
    "F-statistic": [f_stat],
    "p-value": [p_val]
})

# Define the output directory based on the specified path
output_dir = Path.home()/"Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel/results"

# Make sure the directory exists
os.makedirs(output_dir, exist_ok=True)
# Debug print to show the directory path
print(f"Output directory: {output_dir}")
# Save the ANOVA results
anova_results_path = output_dir / "anova_results.tsv"
anova_df.to_csv(anova_results_path, sep="\t", index=False)
print("ANOVA results saved : {anova_results_path}")

# 4. Create a boxplot for the three treatments

# plt.figure(figsize=(10, 6))
# sns.set(style="whitegrid")  # Nice background style

# # Boxplot with color palette
# sns.boxplot(x="Treatment", y="Displacement", data=df_long, palette="pastel", showmeans=True,
#             meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black"})

# # Titles and labels
# plt.title("Mean Displacement by Gradients", fontsize=16, weight='bold')
# plt.xlabel("Gradient Type", fontsize=12)
# plt.ylabel("Mean Displacement", fontsize=12)

# # Save the enhanced plot
# # Save the enhanced plot - use absolute path format and explicitly close the figure
# boxplot_path = os.path.join(str(output_dir), "anova_boxplot.png")
# plt.savefig(boxplot_path)
# plt.close()

# print(f"Boxplot saved to: {boxplot_path}")
#####################-new box plot here
# 4. Create an enhanced boxplot for the three treatments
plt.figure(figsize=(12, 8))

# Set a more professional theme
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

# Custom color palette
custom_palette = sns.color_palette("viridis", 3)

# Enhanced boxplot
ax = sns.boxplot(x="Treatment", y="Displacement", data=df_long, 
                 palette=custom_palette, width=0.5,
                 linewidth=1.5, showmeans=True, 
                 meanprops={"marker":"D", "markerfacecolor":"white", 
                           "markeredgecolor":"black", "markersize":8})

# Add individual data points with jitter
sns.stripplot(x="Treatment", y="Displacement", data=df_long,
              jitter=True, size=5, alpha=0.5, color="black")

# Add statistical annotation if desired
if p_val < 0.05:
    plt.text(0.5, df_long["Displacement"].max() * 1.05, 
             f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}*", 
             ha='center', fontsize=12, color='red')
else:
    plt.text(0.5, df_long["Displacement"].max() * 1.05, 
             f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}", 
             ha='center', fontsize=12)

# Enhance aesthetics
plt.title("Mean Displacement by Gradient Type", fontsize=18, weight='bold', pad=20)
plt.xlabel("Gradient Pattern", fontsize=16, weight='bold', labelpad=15)
plt.ylabel("Mean Displacement (units)", fontsize=16, weight='bold', labelpad=15)

# Add grid just to the y-axis for cleaner look
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=14)

# Tight layout
plt.tight_layout()

# Add a subtle border
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
    
# Save the enhanced plot with higher DPI for publication quality
boxplot_path = os.path.join(str(output_dir), "anova_boxplot_enhanced.svg")
plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Enhanced boxplot saved to: {boxplot_path}")
#################################
#swarm plot
# plt.figure(figsize=(10, 6))
# # sns.swarmplot(x='source_folder', y='Displacement', data=df_filtered, hue='source_folder', palette='viridis', size=2, alpha=0.7)
# sns.stripplot(x='Treatment', y='Displacement', data=df_long, hue='Treatment', palette='magma', size=4, alpha=0.6, jitter=True)

# sns.pointplot(x='Treatment', y='Displacement', data=df_long, color='red', scale=0.5, 
#               markers='d', join=False)
# # Titles and labels
# plt.title("Mean Displacement by Gradients", fontsize=16, weight='bold')
# plt.xlabel("Gradient Type", fontsize=12)
# plt.ylabel("Mean Displacement", fontsize=12)

# plt.grid(True, linestyle='--', alpha=0.7)
# swarm_plot_path = os.path.join(str(output_dir), "anova_swarm_plots.png")
# plt.savefig(swarm_plot_path)
# plt.close()

# print(f"Swarm plot saved to: {swarm_plot_path}")
#################new plot here

# Enhanced swarm/strip plot
plt.figure(figsize=(12, 8))

# Set theme
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)

# Background color
plt.gcf().set_facecolor('#f8f9fa')

# Create a more visually appealing color palette
custom_palette = sns.color_palette("magma", 3)

# Add violin plot for distribution
sns.violinplot(x='Treatment', y='Displacement', data=df_long, 
               palette=custom_palette, alpha=0.3, inner=None)

# Add individual data points
sns.stripplot(x='Treatment', y='Displacement', data=df_long, 
              size=8, alpha=0.7, jitter=True, 
              palette=custom_palette, linewidth=1, edgecolor='black')

# Add mean points
sns.pointplot(x='Treatment', y='Displacement', data=df_long, 
              color='black', scale=1.2, markers='D', join=False, 
              errorbar=('ci', 95), errwidth=2)

# Add statistical annotation
if p_val < 0.05:
    plt.annotate(f"ANOVA: p={p_val:.4f}*", 
                xy=(0.5, 0.97), xycoords='figure fraction',
                fontsize=12, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# Enhanced styling
plt.title("Cellular Displacement in Different Gradient Environments", 
          fontsize=18, weight='bold', pad=20)
plt.xlabel("Gradient Type", fontsize=16, weight='bold', labelpad=15)
plt.ylabel("Mean Displacement", fontsize=16, weight='bold', labelpad=15)

# Remove legend as color already differentiates treatments
plt.legend([],[], frameon=False)

# Add subtle horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Add a subtle border
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    
plt.tight_layout()

# Save high resolution
swarm_plot_path = os.path.join(str(output_dir), "anova_swarm_plots_enhanced.png")
plt.savefig(swarm_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Enhanced swarm plot saved to: {swarm_plot_path}")



# 5. If the ANOVA result is significant, perform Tukey HSD test
alpha = 0.05
if p_val < alpha:
    print("ANOVA is significant (p < 0.05). Running Tukey HSD post-hoc test.")
    tukey = pairwise_tukeyhsd(endog=df_long["Displacement"],
                              groups=df_long["Treatment"],
                              alpha=alpha)
    
    # Convert Tukey summary table to a DataFrame and save it
    # The first row of data is a header row in the summary table.
    summary_data = tukey.summary().data
    tukey_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    tukey_path = os.path.join(str(output_dir), "tukey_results.tsv")
    tukey_df.to_csv(tukey_path, sep="\t", index=False)
    print(f"Tukey test results saved to: {tukey_path}")
    
    
    # Plot the Tukey HSD results
    # fig = tukey.plot_simultaneous(comparison_name="Linear")
    # plt.title("Tukey HSD Test Comparisons")
    # plt.xlabel("Mean Difference")
    # tukey_plot_path = os.path.join(str(output_dir), "tukey_plot.png")
    # plt.savefig(tukey_plot_path)
    # plt.close()
    # print(f"Tukey plot saved to: {tukey_plot_path}")
    ########new plot here
    # Only runs if ANOVA is significant
    # Better Tukey HSD visualization

    # Alternative custom Tukey HSD visualization
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    # Extract data from tukey_df
    comparisons = []
    mean_diffs = []
    conf_intervals = []
    pvalues = []
    
    for i, row in tukey_df.iterrows():
        comparisons.append(f"{row['group1']} vs {row['group2']}")
        mean_diffs.append(row['meandiff'])
        conf_intervals.append([row['lower'], row['upper']])
        pvalues.append(row['p-adj'])
    
    # Convert to numpy arrays
    mean_diffs = np.array(mean_diffs)
    conf_intervals = np.array(conf_intervals)
    
    # Sort by mean difference
    idx = np.argsort(mean_diffs)
    comparisons = [comparisons[i] for i in idx]
    mean_diffs = mean_diffs[idx]
    conf_intervals = conf_intervals[idx]
    pvalues = [pvalues[i] for i in idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean differences
    y_positions = np.arange(len(comparisons))
    ax.scatter(mean_diffs, y_positions, s=80, color='blue', zorder=3)
    
    # Plot confidence intervals
    for i, (low, high) in enumerate(conf_intervals):
        ax.plot([low, high], [y_positions[i], y_positions[i]], color='blue', linewidth=2)
        
    # Add vertical line at zero
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add significance markers
    for i, pval in enumerate(pvalues):
        sig_marker = ""
        if pval < 0.001:
            sig_marker = "***"
        elif pval < 0.01:
            sig_marker = "**"
        elif pval < 0.05:
            sig_marker = "*"
            
        if sig_marker:
            ax.text(max(conf_intervals[i]) + 0.02, y_positions[i], sig_marker, 
                   fontsize=14, color='red', va='center')
    
    # Labels and formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(comparisons, fontsize=12)
    ax.set_xlabel('Mean Difference', fontsize=14, fontweight='bold')
    ax.set_title('Tukey HSD Pairwise Comparisons', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add p-value legend
    ax.text(0.95, 0.05, 
           "* p<0.05\n** p<0.01\n*** p<0.001", 
           transform=ax.transAxes, fontsize=12, 
           va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    custom_tukey_path = os.path.join(str(output_dir), "custom_tukey_plot.svg")
    plt.savefig(custom_tukey_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Custom Tukey plot saved to: {custom_tukey_path}")
    # print(f"Enhanced Tukey plot saved to: {tukey_plot_path}")
else:
    print("ANOVA is not significant (p >= 0.05). Tukey HSD test was not performed.")
