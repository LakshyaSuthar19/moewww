import fitz


pdf_path = "test.pdf"


doc = fitz.open(pdf_path)


for page_number, page in enumerate(doc):

    text = page.get_text().strip()

    print(
        f"\nPAGE {page_number + 1}"
    )

    if len(text) < 10:

        print("OCR required...")

        text_page = page.get_textpage_ocr(
            language="eng",
            dpi=200
        )

        text = page.get_text(
            textpage=text_page
        )

    print(text[:1000])


doc.close()