'''
  Upload files to S3, to be downloaded by the app
'''
import sys, os
import _config, util, string
import boto3

s3 = boto3.client('s3', 
                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

D_FOLD = _config.OUT_PLACE + 'd_annotate/'


S3_SAFE_CHARS = string.ascii_letters + string.digits + "!-_.*'()"
REPLACE_CHAR = '_'
def s3_safe(name):
  convert_safe = lambda char: char if char in S3_SAFE_CHARS else REPLACE_CHAR
  return ''.join(convert_safe(char) for char in name)

'''
  Upload stepchart pkls
'''
def upload_all():
  # Upload e_struct files per stepchart
  files = [fn for fn in os.listdir(D_FOLD) if '.csv' in fn]
  print(f'Uploading {len(files)} d_annotate files ...')
  timer = util.Timer(total=len(files))
  for fn in files:
    s3.upload_file(D_FOLD + fn, os.environ['S3_BUCKET_NAME'], 'd_annotate/' + s3_safe(fn))
    timer.update()

  print(f'Done')
  return


def upload_single():
  nm = 'Headless Chicken - r300k D21 arcade'
  # nm = 'Trashy Innocence - Last Note. D15 arcade'
  fn = nm + '.pkl'
  s3.upload_file(D_FOLD + fn, os.environ['S3_BUCKET_NAME'], s3_safe(fn))
  return


if __name__ == '__main__':
  if sys.argv[1] == 'all':
    upload_all()
  elif sys.argv[1] == 'single':
    upload_single()