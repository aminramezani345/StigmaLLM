import sys
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyPDF2 import PdfReader

class StigmaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Stigma Search GUI')
        self.setGeometry(100, 100, 800, 600)

        self.text_edit = QTextEdit(self)
        self.upload_button = QPushButton('Upload File', self)
        self.upload_button.clicked.connect(self.upload_file)

        self.search_button = QPushButton('Search Stigma', self)
        self.search_button.clicked.connect(self.search_stigma)

        self.results_label = QLabel('Results:', self)
        self.results_text = QTextEdit(self)
        self.results_text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.results_label)
        layout.addWidget(self.results_text)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open PDF File', '', 'PDF Files (*.pdf)', options=options)
        if file_path:
            self.extract_text_from_pdf(file_path)

    def extract_text_from_pdf(self, file_path):
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            self.text_edit.setPlainText(text)

    def search_stigma(self):
        text = self.text_edit.toPlainText()
        if not text:
            self.results_text.setPlainText("No text provided for search.")
            return
        response = requests.post('http://127.0.0.1:5000/process', json={'text': text})
        if response.status_code == 200:
            results = response.json().get('results', [])
            self.results_text.setPlainText("\n".join(str(result) for result in results))
        else:
            self.results_text.setPlainText(f"Error: {response.status_code}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StigmaApp()
    window.show()
    sys.exit(app.exec_())
