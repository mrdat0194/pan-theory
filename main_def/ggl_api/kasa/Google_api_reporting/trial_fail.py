from googleapiclient import discovery, errors
from oauth2client import file, client, tools
from google.oauth2 import service_account
from email.mime.text import MIMEText
import base64

SERVICE_ACCOUNT_FILE = 'shopkasatriavn-556c77980273.json'
SCOPES = [  'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.send',
            'https://www.googleapis.com/auth/gmail.settings.sharing',
            'https://www.googleapis.com/auth/gmail.settings.basic'
          ]

# The user we want to "impersonate"
# USER_EMAIL = "supportvn@kasatria.com"
USER_EMAIL = "ga4-report-sender@shopkasatriavn.iam.gserviceaccount.com"
# USER_EMAIL = "source@domain.com"
def validationService():
    # Set the crendentials
    credentials = service_account.Credentials.\
        from_service_account_file(SERVICE_ACCOUNT_FILE)
    creds = credentials.with_scopes(SCOPES)
    # Delegate the credentials to the user you want to impersonate
    delegated_credentials = creds.with_subject(USER_EMAIL)
    # creds = creds.create_delegated(user_email)
    service = discovery.build('gmail', 'v1', credentials=delegated_credentials)
    return service


def SendMessage(service, message):
    message = service.users().messages().send(userId="me", body=message).execute()
    return message


def CreateMessage(sender, to, subject, message_text):
  message = MIMEText(message_text)
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def main():
    try:
        service = validationService()
        email = CreateMessage(USER_EMAIL, "dat@kasatria.com", "Test", "This is a test")
        print(email)
        email_sent = SendMessage(service, email)
        print('Message Id:', email_sent['id'])
    except errors.HttpError as err:
        print('\n---------------You have the following error-------------')
        print(err)
        print('---------------You have the following error-------------\n')


if __name__ == '__main__':
    main()