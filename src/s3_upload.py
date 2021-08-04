'''
  Upload files to S3, to be downloaded by the app
'''
import _config, util, string
import os
import boto3

s3 = boto3.client('s3', 
                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

# Upload single files
single_fns = {
  _config.DATA_DIR + 'charts_all.csv': 'charts_all.csv',
  _config.OUT_PLACE + 'merge_features/features.csv': 'features.csv',
}
for local_fn, filename in single_fns.items():
  print(f'Uploading {filename} ...')
  s3.upload_file(local_fn, os.environ['S3_BUCKET_NAME'], filename)


S3_SAFE_CHARS = string.ascii_letters + string.digits + "!-_.*'()"
REPLACE_CHAR = '_'
def s3_safe(name):
  convert_safe = lambda char: char if char in S3_SAFE_CHARS else REPLACE_CHAR
  return ''.join(convert_safe(char) for char in name)


# Upload e_struct files per stepchart
e_struct_fold = _config.OUT_PLACE + 'e_struct/'
files = [fn for fn in os.listdir(e_struct_fold) if '.pkl' in fn]
print(f'Uploading {len(files)} e_struct files ...')
timer = util.Timer(total=len(files))
for fn in files:
  s3.upload_file(e_struct_fold + fn, os.environ['S3_BUCKET_NAME'], s3_safe(fn))
  timer.update()

print(f'Done')