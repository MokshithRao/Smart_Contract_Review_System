import pdfplumber

from docx import Document



def extract_text(file_path):

    text = ""



    if file_path.endswith(".pdf"):

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:

                    text += extracted + "\n"



    elif file_path.endswith(".docx"):

        doc = Document(file_path)

        for para in doc.paragraphs:

            text += para.text + "\n"



    elif file_path.endswith(".txt"):

        with open(file_path, "r", encoding="utf-8", errors="replace") as text_file:

            text = text_file.read()



    else:

        return ""



    return text







if __name__ == "__main__":

    file_path = "Anil.pdf"

    result = extract_text(file_path)

    print(result)

