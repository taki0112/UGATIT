from os import environ
from os.path import join, dirname
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import boto3


class EmailService(object):
    EMAIL_REGEX = re.compile(
        r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")

    def __init__(self):
        self.ses = boto3.client('ses')

    def is_email(self, candidate):
        is_email = False
        if self.EMAIL_REGEX.match(candidate):
            is_email = True
        return is_email

    def send_email(self, email_addr, image_url):
        email = self.build_email(email_addr, image_url)
        self.ses.send_raw_email(
            RawMessage={'Data': email.as_string()},
            Source=email['From'],
            Destinations=[email['To']]
        )

    def build_email(self, email_addr, image_url):
        email = MIMEMultipart()
        email['Subject'] = 'selfie2anime Complete!'
        email['From'] = environ.get('SENDER_EMAIL')
        email['To'] = email_addr

        email.preamble = 'Multipart message.\n'

        email_body = self.build_email_body(image_url)
        part = MIMEText(email_body, 'html')
        email.attach(part)

        return email

    @staticmethod
    def build_email_body(image_url):
        html_file = join(dirname(__file__),
                         'templates', 'template.html')
        html_file = open(html_file, 'r')
        email = html_file.read()
        email = email.replace('{{image_url}}', image_url)
        return email