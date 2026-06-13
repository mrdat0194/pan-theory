import pytesseract
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output = os.path.join(BASE_DIR, "output_toText.txt")
if os.path.exists(output):
    f = open(output, "r+")
else:
    f = open(output, "w")

output_text = pytesseract.image_to_string('/Users/petern/Desktop/Mypage/pan-theory/test_toText.jpg')
print(output_text)
f.write(output_text)
f.close()
myfile = open(output, 'rb')
print(myfile.readlines())




