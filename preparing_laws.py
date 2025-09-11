
"""
This file is optional to execute.

Each line has a law article number, e.g., "المادة 9".
We attach each line of law text to its article number.

After execution, a new file is generated with the name defined in the env file:
    output_file_dir = path/to/Cybercrime_laws.txt
"""

import os
import re
from dotenv import load_dotenv


def preprocess_laws():
    load_dotenv()
    file_dir = os.getenv("text_file_path")
    output_file_dir = os.getenv("output_file_path")

    if not file_dir or not output_file_dir:
        raise ValueError("Missing text_file_path or output_file_dir in .env file")

    with open(file_dir, "r", encoding="utf-8") as f:
        file_content = f.readlines()

    belongs_to = ""
    with open(output_file_dir, "w", encoding="utf-8") as processed_file:
        for line in file_content:
            if re.match(r"المادة ", line):
                belongs_to = line.strip() + " تنص على "
                continue
            processed_file.write(belongs_to + line)

    print(f"✅ Processed laws saved to {output_file_dir}")