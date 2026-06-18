import json
import os
import re
import tokenize
import io

def remove_comments(source_code):
    io_obj = io.StringIO(source_code)
    out = ""
    last_lineno = -1
    last_col = 0
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            else:
                out += token_string
            last_lineno = end_line
            last_col = end_col
        # Clean up empty lines that consist only of spaces/newlines left by comments
        # But be careful not to remove intentional empty lines
        # Actually, let's just return out and do a simpler blank line cleanup
        return out
    except Exception as e:
        # Fallback if there is a syntax error or tokenization error
        lines = source_code.split('\n')
        out_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            out_lines.append(line)
        return '\n'.join(out_lines)

def clean_notebook(filepath, title, description):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Add professional markdown title cell if the first cell doesn't seem to be a title
    # We will just insert it at the beginning.
    # To avoid duplicates, check if the first cell is already a markdown cell with a `#` title.
    if nb['cells'] and nb['cells'][0]['cell_type'] == 'markdown':
        source = nb['cells'][0]['source']
        if isinstance(source, list) and len(source) > 0 and source[0].startswith('#'):
            # Replace it to be standard and professional
            nb['cells'][0]['source'] = [
                f"# {title}\n",
                "\n",
                f"{description}\n"
            ]
        else:
            # Insert at beginning
            nb['cells'].insert(0, {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    f"{description}\n"
                ]
            })
    else:
        nb['cells'].insert(0, {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                f"{description}\n"
            ]
        })

    # 2. Remove comments from code cells and remove emojis from markdown cells
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code_str = "".join(source)
            else:
                code_str = source
            
            cleaned_code = remove_comments(code_str)
            # Reconstruct list of lines (jupyter format)
            lines = cleaned_code.splitlines(True)
            cell['source'] = lines
            
        elif cell['cell_type'] == 'markdown':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # Remove emojis
                    clean_line = emoji_pattern.sub('', line)
                    new_source.append(clean_line)
                cell['source'] = new_source
            else:
                cell['source'] = emoji_pattern.sub('', source)
                
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')

notebooks_info = [
    {
        "file": r"c:\Users\hegde\Documents\Github\iv-fluids-level-monitor-and-drop-count\notebooks\iv-fluids-drop-count-heatmap.ipynb",
        "title": "IV Fluids Drop Count Heatmap",
        "desc": "This notebook implements a heatmap-based approach for IV fluids drop counting and analysis."
    },
    {
        "file": r"c:\Users\hegde\Documents\Github\iv-fluids-level-monitor-and-drop-count\notebooks\iv-fluids-drop-detection-yolov8.ipynb",
        "title": "IV Fluids Drop Detection Using YOLOv8",
        "desc": "This notebook implements drop detection for IV fluids using the YOLOv8 object detection model."
    },
    {
        "file": r"c:\Users\hegde\Documents\Github\iv-fluids-level-monitor-and-drop-count\notebooks\iv-fluids-level-monitor.ipynb",
        "title": "IV Fluids Level Monitor",
        "desc": "This notebook implements a complete workflow to train a convolutional neural network (CNN) for classifying the fill level of IV fluid bottles."
    }
]

for info in notebooks_info:
    print(f"Processing {info['file']}...")
    clean_notebook(info['file'], info['title'], info['desc'])
print("Done.")
