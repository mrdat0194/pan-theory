from Google import create_service
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import io
import pandas as pd

CLIENT_SECRET_FILE = 'client_secret_499212216558-8bchgfkfrrmrctsej92na04hgqr03nmt.apps.googleusercontent.com'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/','https://www.googleapis.com/auth/gmail.readonly']

service = create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

def df_to_csv(df):
    with io.StringIO() as buffer:
        df.to_csv(buffer)
        return buffer.getvalue()

df = pd.read_csv("GA4_SKY.csv")
print(df)
attachment = MIMEApplication(df_to_csv(df))
attachment['Content-Disposition'] = 'attachment; filename="ga4report.csv"'


emailMsg = 'You won $100,000'
mimeMessage = MIMEMultipart()
mimeMessage['to'] = 'dat@kasatria.com'
mimeMessage['subject'] = 'You won'
mimeMessage.attach(attachment)

# mimeMessage.attach(MIMEText(emailMsg, 'plain'))
raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()
print(raw_string)
message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
print(message)