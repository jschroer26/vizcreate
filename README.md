# vizcreate
# VizCreate

**Natural Language Data Visualization for Education**

VizCreate is an AI-assisted visualization tool that allows users to describe a chart in plain English and automatically generates an appropriate visualization.

Rather than requiring users to understand plotting libraries or complex chart settings, VizCreate interprets the user's request, recognizes the structure of the uploaded dataset, recommends an appropriate visualization, and explains the resulting chart.

Designed originally for educational assessment data (especially Wyoming WYTOPP data), VizCreate also supports many general tabular datasets.

---

## Features

### Natural Language Visualization

Simply describe the chart you want.

Examples:

- Create a line chart showing ELA proficiency over the past five years by grade.
- Compare Math proficiency across grades.
- Create a heatmap of Science proficiency by grade and school year.
- Show the average reading score by gender.

---

### Dataset Recognition

VizCreate automatically identifies common dataset families, including:

- WYTOPP longitudinal assessment summaries
- Student-level assessment datasets
- Survey / Likert data
- General tabular datasets

Visualization recommendations are adjusted automatically for each dataset.

---

### Supported Visualizations

- Bar Charts
- Grouped Bar Charts
- WYTOPP Stacked Performance Charts
- Line Charts
- Heatmaps
- Box-and-Whisker Plots

---

### Interactive Controls

After a chart is generated, users can:

- Filter grade levels
- Filter school years
- Change color palettes
- Override titles
- Override axis labels
- Display value labels
- Display sample sizes (N)
- Download PNG images

---

### AI-Assisted Interpretation

Every visualization automatically includes:

### Visualization Summary

A plain-language explanation describing exactly what the chart displays.

Example:

> This line chart displays the average Percent Proficient and Advanced in English Language Arts across five school years. Each line represents a grade level, allowing comparisons of achievement trends over time.

---

### What to Notice

VizCreate automatically identifies meaningful observations from the visualization, such as:

- largest increases
- largest decreases
- highest proficiency
- lowest proficiency
- strongest trends
- notable comparisons

These observations are generated directly from the displayed data.

---

## Built for Education

VizCreate was developed to help educators, school leaders, and researchers quickly explore educational data without requiring programming knowledge.

Although originally designed around Wyoming's WYTOPP assessment data, VizCreate works with many tabular datasets.

---

## Example Workflow

1. Upload a CSV or Excel file.
2. Describe the visualization in natural language.
3. VizCreate builds the chart.
4. Refine with interactive controls.
5. Read the automatically generated interpretation.
6. Export the visualization.

---

## Technology

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- OpenAI API

---

## Roadmap

Future development includes:

- Scatter plots
- Slopegraphs
- Histograms
- Pie charts
- Interactive Plotly visualizations
- Automatic statistical summaries
- "Interpret This Chart" mode
- "Create This Chart" from uploaded images
- Dashboard generation from multiple prompts
- PDF report generation
- Accessibility improvements

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/VizCreate.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run vizcreate_app.py
```

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify, and distribute this software under the terms of the GPL-3.0 license.

See the LICENSE file for details.

---

## Citation

If you use VizCreate in research, presentations, or publications, please cite the software.

**Suggested citation**

Schroer, J. (2026). *VizCreate: Natural Language Data Visualization for Education* (Version 1.0). University of Wyoming.

---

## Author

**Joseph Schroer, Ph.D.**

College of Education

University of Wyoming

Learning Sciences • Educational Assessment • Artificial Intelligence • Data Visualization

---

*"Helping educators spend less time creating charts and more time understanding learning."*
