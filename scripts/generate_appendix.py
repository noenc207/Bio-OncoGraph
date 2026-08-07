import os
import json

def generate_appendix():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
    os.makedirs(output_dir, exist_ok=True)
    
    content = r"""\appendix
\section{Supplementary Data}
This section contains supplementary data tables and additional experimental metrics that support the main findings of the paper.

\section{Extended Methodology}
Further details regarding the benchmark consistency validation scripts and parameter tuning are provided here.
"""

    filepath = os.path.join(output_dir, "appendix.tex")
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(content)
    
    print(f"Appendix generated successfully at {filepath}")

if __name__ == "__main__":
    generate_appendix()
