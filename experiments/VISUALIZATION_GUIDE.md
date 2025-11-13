# Publication-Quality Visualizations

## Overview

The `analyze_results.py` script generates **6 publication-ready figures** in both **PDF (vector)** and **PNG (raster)** formats, suitable for inclusion in formal scientific papers.

## Key Features

✅ **Publication-ready styling**
- White/transparent backgrounds
- LaTeX rendering (with graceful fallback)
- Colorblind-friendly palettes
- Clean, minimal design (no top/right spines)
- Proper grid placement (behind data)

✅ **Vector and raster formats**
- PDF files: Vector graphics, transparent background
- PNG files: Raster graphics, white background, 300 DPI

✅ **Professional typography**
- Computer Modern Roman font (LaTeX)
- Consistent font sizes across all plots
- Proper mathematical notation

## Generated Figures

### 1. Error Distribution by Dimension
**File:** `error_distribution_by_dimension.{pdf,png}`

**Type:** Violin plot

**Description:** Shows the distribution of log₁₀(error) for each problem dimension (D=2, D=10) using violin plots. Violin plots show:
- Full distribution shape (kernel density estimate)
- Median (horizontal line)
- Interquartile range
- Min/max values

**Key features:**
- Colorblind-friendly blue color
- Transparent violins with black edges
- Mean and median markers
- Grid for easy reading

**Use in paper:** Demonstrates how error distributions differ between 2D and 10D problems.

---

### 2. Error by Function
**File:** `error_by_function.{pdf,png}`

**Type:** Grouped bar chart with error bars

**Description:** Shows mean log₁₀(error) for each function ID, grouped by dimension. Error bars represent standard deviation across instances.

**Key features:**
- Two bars per function (D=2 in blue, D=10 in orange)
- Error bars showing variability
- Horizontal grid for reference
- Compact legend

**Use in paper:** Comparison of optimizer performance across different benchmark functions.

---

### 3. Operator Usage by Dimension
**File:** `operator_usage_by_dimension.{pdf,png}`

**Type:** Stacked bar chart

**Description:** Shows the percentage usage of each operator (LF, DM, PS, SP) for each dimension, stacked to 100%.

**Key features:**
- Colorblind-friendly palette:
  - LF (Lévy Flight): Orange (#E69F00)
  - DM (Differential Mutation): Sky Blue (#56B4E9)
  - PS (Particle Swarm): Bluish Green (#009E73)
  - SP (Spiral): Vermillion (#D55E00)
- Black edge lines for clarity
- Legend with operator names

**Use in paper:** Illustrates how the basal ganglia adapts operator selection based on problem dimensionality.

---

### 4. Error vs Operator Usage
**File:** `error_vs_operator_usage.{pdf,png}`

**Type:** 2×2 scatter plots with regression lines

**Description:** Four scatter plots showing the correlation between operator usage percentage and optimization error. Each subplot corresponds to one operator.

**Key features:**
- Scatter points with transparency
- Linear regression line (dashed black)
- Pearson correlation coefficient displayed in box
- Consistent color scheme per operator
- Grid for readability

**Use in paper:** Statistical analysis of which operators correlate with better/worse performance.

---

### 5. Error Violin by Function
**File:** `error_violin_by_function.{pdf,png}`

**Type:** Violin plots

**Description:** Shows error distributions for each function ID using violin plots. Better than boxplots for showing multimodal distributions.

**Key features:**
- Green violins (#009E73) with transparency
- Median line (black, thick)
- Interquartile range markers (black vertical lines)
- Wide format to accommodate 9 functions

**Use in paper:** Detailed performance analysis showing full error distributions per function, revealing multimodality and outliers.

---

### 6. Performance Heatmap
**File:** `performance_heatmap.{pdf,png}`

**Type:** Heatmap with annotations

**Description:** Matrix showing median log₁₀(error) for each combination of function ID (rows) and dimension (columns).

**Key features:**
- Red-Yellow-Green diverging colormap (reversed)
  - Red: High error (poor performance)
  - Yellow: Medium error
  - Green: Low error (good performance)
- Numerical annotations in each cell
- Colorbar with clear label

**Use in paper:** Compact overview of performance across all experimental conditions, ideal for at-a-glance comparison.

---

## Technical Specifications

### Font Sizes
- Main text: 11 pt
- Axis labels: 12 pt
- Axis titles: 13 pt
- Tick labels: 10 pt
- Legend: 10 pt

### Dimensions
- Standard figures: 7-12 inches width × 5-8 inches height
- Wide figures: 10-14 inches width (for many categories)
- DPI: 300 (publication quality)

### Color Palette (Colorblind-Friendly)
Based on Wong (2011) palette for scientific visualization:
```python
{
    'LF': '#E69F00',  # Orange
    'DM': '#56B4E9',  # Sky Blue
    'PS': '#009E73',  # Bluish Green
    'SP': '#D55E00',  # Vermillion
}
```

This palette is:
- Distinguishable by people with deuteranopia and protanopia
- Print-friendly (works in grayscale)
- Screen-friendly (good contrast)

### LaTeX Rendering
- Automatic detection and fallback
- Uses Computer Modern Roman font (standard LaTeX)
- Math mode for symbols: $\log_{10}(\mathrm{Error})$
- Gracefully degrades to DejaVu Serif if LaTeX unavailable

## Usage in LaTeX Documents

### Including PDF (vector graphics - recommended)
```latex
\usepackage{graphicx}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{error_distribution_by_dimension.pdf}
    \caption{Error distribution across problem dimensions.}
    \label{fig:error_dist}
\end{figure}
```

### Including PNG (raster graphics)
```latex
\usepackage{graphicx}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{error_distribution_by_dimension.png}
    \caption{Error distribution across problem dimensions.}
    \label{fig:error_dist}
\end{figure}
```

### Two-column layout
```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{error_by_function.pdf}
    \caption{Performance comparison across benchmark functions.}
    \label{fig:error_func}
\end{figure*}
```

### Subfigures
```latex
\usepackage{subcaption}

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{error_distribution_by_dimension.pdf}
        \caption{Error distributions}
        \label{fig:dist}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{operator_usage_by_dimension.pdf}
        \caption{Operator usage}
        \label{fig:usage}
    \end{subfigure}
    \caption{Performance analysis by dimension}
\end{figure}
```

## Recommended Figures for Different Paper Sections

### Results Section
1. **Error Violin by Function** - Main performance results
2. **Error by Function** - Mean comparison with variability
3. **Performance Heatmap** - Compact overview

### Analysis Section
4. **Error Distribution by Dimension** - Scaling analysis
5. **Operator Usage by Dimension** - Algorithm behavior
6. **Error vs Operator Usage** - Correlation analysis

## Customization

To modify the plots, edit `analyze_results.py`:

### Change colors
```python
op_colors = {
    'LF': '#YOUR_COLOR',  # Hex color code
    'DM': '#YOUR_COLOR',
    'PS': '#YOUR_COLOR',
    'SP': '#YOUR_COLOR',
}
```

### Change figure size
```python
fig, ax = plt.subplots(figsize=(width, height))  # in inches
```

### Change DPI
```python
plt.rcParams['savefig.dpi'] = 600  # Higher resolution
```

### Disable LaTeX
```python
use_latex = False  # Force disable at line ~145
```

## Quality Checklist

Before submitting to a journal:

✅ Figures are saved in vector format (PDF preferred)  
✅ DPI is at least 300 for raster images  
✅ Text is readable at final publication size  
✅ Colors are distinguishable in grayscale  
✅ Axis labels include units/scales  
✅ Legend is clear and positioned well  
✅ Grid lines don't obscure data  
✅ No unnecessary elements (removed spines)  
✅ Consistent style across all figures  
✅ File sizes are reasonable (<5MB per figure)  

## References

- Wong, B. (2011). "Points of view: Color blindness." *Nature Methods* 8(6): 441.
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information*. Graphics Press.
- Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment." *Computing in Science & Engineering* 9(3): 90-95.

## Troubleshooting

### "LaTeX not available" warning
**Cause:** LaTeX not installed or not in PATH  
**Solution:** 
- Install MacTeX: `brew install --cask mactex-no-gui`
- Or use standard fonts (automatic fallback)

### Figures look pixelated
**Cause:** Using PNG at too small size  
**Solution:** Use PDF files (vector graphics) or increase DPI to 600

### Text too small in final document
**Cause:** Figure scaled down too much  
**Solution:** Use `figsize=(larger, values)` in script or increase `\textwidth` fraction in LaTeX

### Colors don't match between figures
**Cause:** Inconsistent color definitions  
**Solution:** All colors defined in `op_colors` dictionary - edit once, applies everywhere

---

**Generated by:** `analyze_results.py`  
**Last updated:** 2025-11-13

