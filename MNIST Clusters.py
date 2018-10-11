
# coding: utf-8

# # MNIST Clusters

# In[5]:


from sagemaker import get_execution_role
role = get_execution_role()


# In[6]:


get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)

def show_digit(img, caption='', subplot=None):
    if subplot == None:
        _, (subplot) = plt.subplots(1,1)
    imgr = img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# In[4]:


get_ipython().run_cell_magic('time', '', "import io\nimport boto3\nfrom sagemaker.amazon.common import write_numpy_to_dense_tensor\n\nbucket = '2018-10-08-batch-test' # Use the name of your s3 bucket here\ntrain_folder = 'KMtest'\ntest_folder = 'KMtest'\nmodel_location = 'KMtest'\n\ntrainURL = 's3://{}/{}'.format(bucket, train_folder)\ntestURL = 's3://{}/{}'.format(bucket, test_folder)\nmodelFolder = 's3://{}/{}'.format(bucket, model_location)\nprint('training data will be uploaded to: {}'.format(trainURL))\nprint('training artifacts will be created in: {}'.format(modelFolder))\n\n# Convert the training data into the format required by the SageMaker KMeans algorithm\nbuf = io.BytesIO()\nwrite_numpy_to_dense_tensor(buf, train_set[0], train_set[1])\nbuf.seek(0)\n\nboto3.resource('s3').Bucket(bucket).Object(train_folder).upload_fileobj(buf)")


# In[ ]:


# from time import gmtime, strftime
# job_name = 'KMeans-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) 
# print("Training job", job_name)


# In[ ]:


from sagemaker import KMeans

kmeans = KMeans(role=role,
                train_instance_count=2,
                train_instance_type='ml.c4.8xlarge',
                output_path="s3://2018-10-08-batch-test",
                k=10,
                data_location=trainURL)


# Use the high-level SDK

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nkmeans.fit(kmeans.record_set(train_set[0]))')


# In[ ]:


kmeans.latest_training_job.job_name
modelURL = 's3://{}/{}/output/model.tar.gz'.format(bucket, kmeans.latest_training_job.job_name)
modelURL


# In[ ]:



sagemaker = boto3.client('sagemaker')
from sagemaker.amazon.amazon_estimator import get_image_uri
image = get_image_uri(boto3.Session().region_name, 'kmeans')

kmeans_hosting_container = {
    'Image': image,
    'ModelDataUrl': modelURL
}


# In[ ]:


kmeans_hosting_container


# In[ ]:


create_model_response = sagemaker.create_model(
    ModelName="MNIST-high-level",
    ExecutionRoleArn=role,
    PrimaryContainer=kmeans_hosting_container)


# In[ ]:


import sagemaker

val_key = 'kmeans_highlevel_example/data/val.data'
prediction_key = 'kmeans_highlevel_example/prediction/valid-data.csv.out'

### Convert the validation set numpy array to a csv file and upload to s3
numpy.savetxt('valid-data.csv', valid_set[0], delimiter=',', fmt='%g')
s3_client = boto3.client('s3')
result = s3_client.upload_file('valid-data.csv', bucket, val_key)
result

inputURL = 's3://{}/{}'.format(bucket, val_key)
outputURL = 's3://{}/{}'.format(bucket, prediction_key)


# Initialize the transformer object
transformer =sagemaker.transformer.Transformer(
    base_transform_job_name='Batch-Transform',
    model_name="MNIST-high-level",
    instance_count=1,
    instance_type='ml.c4.xlarge',
    output_path=outputURL
    )

# To start a transform job:
transformer.transform(inputURL, content_type='text/csv', split_type='Line')

# Then wait until transform job is completed
transformer.wait()

# To fetch validation result 
s3_client.download_file(bucket, prediction_key, 'valid-result')
with open('valid-result') as f:
    results = f.readlines()   
print("Sample transform result: {}".format(results[0]))

