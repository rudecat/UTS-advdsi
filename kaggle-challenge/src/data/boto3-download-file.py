import boto3

s3 = boto3.client('s3')
s3.download_file('uts-advdsi', 'kaggle-nba/sample_submission.csv', 'sample_submission.csv')