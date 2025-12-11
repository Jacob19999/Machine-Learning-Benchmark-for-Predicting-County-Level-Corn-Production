import PyPDF2

pdf = open('From Satellite To Silos Final.pdf', 'rb')
reader = PyPDF2.PdfReader(pdf)
text = ''
for page in reader.pages:
    text += page.extract_text() + '\n'

print(text)
pdf.close()
