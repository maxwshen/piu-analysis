'''
  Upload files to S3, to be downloaded by the app
'''
import _config, util
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


# Upload e_struct files per stepchart
e_struct_fold = _config.OUT_PLACE + 'e_struct/'
files = [fn for fn in os.listdir(e_struct_fold) if '.pkl' in fn]
print(f'Uploading {len(files)} e_struct files ...')
timer = util.Timer(total=len(files))
for fn in files:
  s3.upload_file(e_struct_fold + fn, os.environ['S3_BUCKET_NAME'], fn)
  timer.update()

print(f'Done')