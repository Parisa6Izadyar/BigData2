"""Execute the notebook with this interpreter and a workspace-local kernel."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    output = ROOT / "results/notebook_check"
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "kernel"
    search_path = str(prefix / "share/jupyter")
    os.environ["JUPYTER_PATH"] = search_path + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["ARNO_DEMO"] = "1" if args.demo else "0"
    install(prefix=str(prefix), kernel_name="arno-check", display_name="Arno notebook check")
    notebook = nbformat.read(ROOT / "notebooks/H&S_SARIMAX.ipynb", as_version=4)
    print("Executing notebook on " + ("synthetic data" if args.demo else "the supplied workbook"), flush=True)
    client = NotebookClient(notebook, kernel_name="arno-check", timeout=600,
                            resources={"metadata": {"path": str(ROOT)}},
                            on_cell_start=lambda cell, cell_index: print(f"Cell {cell_index + 1}/{len(notebook.cells)}", flush=True))
    client.execute()
    nbformat.write(notebook, output / "executed.ipynb")
    print(f"Executed all {len(notebook.cells)} notebook cells. Output: {output}")


if __name__ == "__main__":
    main()
