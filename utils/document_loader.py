import PyPDF2
import docx  # For reading DOCX files
import os

def load_pdf(file_path):
    """Loads the content of a document from the given file path."""
    file_extension = os.path.splitext(file_path)[1].lower()

    text = ""

    try:
        if file_extension == ".pdf":
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"

        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        text = f"Error reading the file: {e}"

    return text