"""
Test script for the unstructured library: partition PDFs from the files/ directory.
Place PDF files in the 'files' folder next to this script to process them.
"""

from pathlib import Path

from unstructured.partition.auto import partition


def get_pdf_paths(files_dir: Path) -> list[Path]:
    """Return paths to all PDF files in the given directory."""
    if not files_dir.is_dir():
        return []
    return sorted(files_dir.glob("*.pdf"))


def main() -> None:
    # Resolve files directory relative to this script
    script_dir = Path(__file__).resolve().parent
    files_dir = script_dir / "files"

    pdf_paths = get_pdf_paths(files_dir)

    if not pdf_paths:
        print(f"No PDFs found in {files_dir}")
        print("Add one or more .pdf files to the 'files' directory and run again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s). Partitioning with unstructured...\n")

    for path in pdf_paths:
        print(f"--- {path.name} ---")
        try:
            elements = partition(filename=str(path))
            print(f"  Elements: {len(elements)}")
            for i, el in enumerate(elements[:10]):  # show first 10
                text_preview = (el.text or "")[:60].replace("\n", " ")
                if len((el.text or "")) > 60:
                    text_preview += "..."
                print(f"    [{i + 1}] {type(el).__name__}: {text_preview}")
            if len(elements) > 10:
                print(f"    ... and {len(elements) - 10} more")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
