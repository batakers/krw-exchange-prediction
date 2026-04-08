import nbformat
import sys

def summarize_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        for i, cell in enumerate(nb.cells):
            print(f"--- Cell {i} ({cell.cell_type}) ---")
            source = cell.source
            if len(source) > 500:
                print(source[:500] + "\n...[TRUNCATED]...")
            else:
                print(source)
            print()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    summarize_notebook('c:/Team1_Project/Project_Code.ipynb')
