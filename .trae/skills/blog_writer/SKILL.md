---
name: "blog_writer"
description: "Generates high-quality, long-form technical blog posts (10,000+ words) with professional charts and system comparisons. Invoke when user asks to write or refine a blog post."
---

# Blog Writer

This skill ensures that all technical blog posts meet the highest industrial and academic standards.

## When to Invoke
-   When the user asks to write a new blog post based on a paper or technical topic.
-   When refining an existing blog post to meet length or quality standards.
-   When converting external documents (e.g., Lark Docs) into formal blog posts.

## Core Requirements

### 1. Word Count & Depth
-   **Minimum 10,000 characters**: Every post must be a deep dive.
-   **Inclusive Count**: Includes main body, chart descriptions, and references.

### 2. Structured Layout
-   **Point-by-Point + Lists**: Use a structured approach for clarity.
-   **Clear Hierarchy**: Use H2-H4 headers effectively.
-   **Bullet Points**: Highlight key conclusions and takeaways.

### 3. High-Quality Visuals (5+ Charts)
-   **Minimum 5 Professional Charts**: Use Mermaid.js for flowcharts, architecture diagrams, etc.
-   **Detailed Descriptions**: Each chart must have **200+ characters** of explanatory text.
-   **Attribution**: Ensure charts are either original (Mermaid) or properly cited.

### 4. Systemic Comparison
-   **Methodology Contrast**: Compare the main topic with at least 2 alternative methods.
-   **Comparison Table**: Create a table with at least **3 dimensions**.
-   **Pros/Cons**: List at least **3 advantages** and **3 limitations** for each method.

### 5. Quality Control Workflow
-   **Triple-Pass Audit**:
    1.  **Technical Accuracy**: Verify all math, formulas, and architecture details.
    2.  **Logical Flow**: Ensure smooth transitions between sections.
    3.  **Format Compliance**: Check headers, list formatting, and chart rendering.
-   **Final Validation**: Ensure all Mermaid diagrams and MathJax formulas render correctly.

## Writing Style
-   **Professional & Educational**: Balance deep technical insight with clear explanations.
-   **Active Voice**: Use present tense and active voice for better readability.
-   **Context-Aware**: Reference previous posts or foundational concepts when relevant.
