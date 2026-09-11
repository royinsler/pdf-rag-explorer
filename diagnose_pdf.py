"""
Standalone diagnostic — not part of the app. Compares text extraction across
a few PDF loaders on one file, to figure out which one (if any) correctly
decodes it. Useful for PDFs whose extracted text looks garbled/reversed
(mojibake) despite the file displaying correctly — usually a sign that the
PDF's embedded font has no reliable ToUnicode CMap for the loader to use.

Usage:
    python diagnose_pdf.py "/path/to/file.pdf"
"""
import sys


def show(label, text):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    if text is None:
        print("(loader not available — see note above)")
        return
    snippet = text[:400].replace("\n", " | ")
    print(snippet)
    print(f"\n[first 60 char codepoints]: {[hex(ord(c)) for c in text[:60]]}")


def try_pdfminer(path):
    from langchain_community.document_loaders import PDFMinerLoader
    docs = PDFMinerLoader(path).load()
    return docs[0].page_content if docs else ""


def try_pypdf(path):
    from langchain_community.document_loaders import PyPDFLoader
    docs = PyPDFLoader(path).load()
    return docs[0].page_content if docs else ""


def try_pymupdf(path):
    from langchain_community.document_loaders import PyMuPDFLoader
    docs = PyMuPDFLoader(path).load()
    return docs[0].page_content if docs else ""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_pdf.py <path-to-pdf>")
        sys.exit(1)

    path = sys.argv[1]

    for name, fn in [
        ("PDFMinerLoader (currently used in app.py)", try_pdfminer),
        ("PyPDFLoader (pypdf — already in requirements.txt)", try_pypdf),
        ("PyMuPDFLoader (needs: pip install pymupdf)", try_pymupdf),
    ]:
        try:
            text = fn(path)
            show(name, text)
        except ImportError as e:
            show(name, None)
            print(f"  -> {e}")
        except Exception as e:
            show(name, None)
            print(f"  -> ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Whichever loader above shows readable Hebrew (not scrambled")
    print("Latin/number characters) is the one we should switch app.py to.")
