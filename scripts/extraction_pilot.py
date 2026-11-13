from __future__ import annotations

import io
import json
import shutil
import subprocess
import time
from pathlib import Path
from importlib.metadata import version

from assistant.artifacts import atomic_json
from assistant.config import PROJECT_ROOT


def main():
    from pypdf import PdfReader,PdfWriter
    from markitdown import MarkItDown
    selected = [("2307.03172",1),("2307.03172",3),("2005.11401",2),("2005.11401",3),("2309.15217",2),("2005.11401",6)]
    papers = {p["paper_id"]:p for p in json.loads((PROJECT_ROOT/"data/corpora/course_rag_v1.source.json").read_text())["papers"]}
    folder = PROJECT_ROOT/"results/extraction-pilot"
    folder.mkdir(parents=True,exist_ok=True)
    converter = MarkItDown()
    rows = []
    for key,number in selected:
        source = papers[key]
        page = PdfReader(source["pdf_path"]).pages[number-1]
        start = time.perf_counter()
        plain = page.extract_text() or ""
        plain_time = time.perf_counter()-start
        writer = PdfWriter(); writer.add_page(page)
        stream = io.BytesIO(); writer.write(stream); stream.seek(0)
        start = time.perf_counter()
        markdown = converter.convert_stream(stream,file_extension=".pdf").markdown
        markdown_time = time.perf_counter()-start
        stem = f"{key}-p{number}"
        (folder/f"{stem}.pypdf.txt").write_text(plain,encoding="utf-8")
        (folder/f"{stem}.markitdown.md").write_text(markdown,encoding="utf-8")
        poppler = shutil.which("pdftoppm.exe") or shutil.which("pdftoppm")
        if not poppler:
            raise RuntimeError("Poppler is required for this visual extraction pilot")
        subprocess.run([poppler,"-f",str(number),"-l",str(number),"-singlefile","-scale-to","1250","-png",
                        source["pdf_path"],str(folder/stem)],check=True,capture_output=True)
        rows.append({"paper_id":key,"page_number":number,"pypdf_characters":len(plain),
            "markitdown_characters":len(markdown),"pypdf_seconds":plain_time,"markitdown_seconds":markdown_time,
            "preview":str(folder/f"{stem}.png"),"review_status":"pending_visual_and_source_anchor_review"})
        print(f"Converted and rendered {stem}",flush=True)
    atomic_json(folder/"measurements.json",{"versions":{name:version(name) for name in ["pypdf","markitdown"]},"pages":rows})


if __name__ == "__main__":
    main()
