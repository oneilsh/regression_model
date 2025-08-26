# Desired API

This repo provides an opinioned framework for rapidly training Cox proportional hazards models on AllOfUs
data, using a configuration .yaml file to specify outcomes and predictors.

This would normally be done within a jupyter notebook in the AoU environment, which comes with a large set of pre-installed python packages:

absl-py==1.4.0
aiodns==2.0.0
aiofiles==22.1.0
aiohttp==3.9.5
aiohttp-cors==0.7.0
aiosignal==1.3.2
aiosqlite==0.20.0
ansicolors==1.1.8
anyio==3.7.1
apache-beam==2.46.0
appdirs==1.4.4
archspec==0.2.3
argcomplete==3.6.2
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
array-record==0.6.0
arrow==1.3.0
arviz==0.21.0
astroid==3.3.9
asttokens==3.0.0
astunparse==1.6.3
async-timeout==4.0.3
atpublic==4.1.0
attrs==24.3.0
avro==1.11.3
azure-common==1.1.28
azure-core==1.33.0
azure-identity==1.21.0
azure-mgmt-core==1.5.0
azure-mgmt-storage==20.1.0
azure-storage-blob==12.25.1
babel==2.16.0
backports.tarfile==1.2.0
beatrix-jupyterlab==2024.920.84202
beautifulsoup4==4.12.3
bgzip==0.3.5
bidict==0.23.1
bigframes==0.22.0
biopython==1.85
bleach==6.0.0
blessed==1.20.0
bokeh==3.4.3
boltons==24.0.0
boto3==1.38.6
botocore==1.38.6
brewer2mpl==1.4.1
Brotli==1.1.0
bx-python==0.13.0
cachecontrol==0.14.2
cached-property==2.0.1
cachetools==4.2.4
certifi==2024.12.14
cffi==1.17.1
charset-normalizer==3.4.0
cli-builder==0.1.5
click==8.1.8
click-plugins==1.1.1
cligj==0.7.2
cloud-tpu-client==0.10
cloud-tpu-profiler==2.4.0
cloudpickle==2.2.1
colorama==0.4.6
coloredlogs==15.0.1
colorful==0.5.6
comm==0.2.2
commonmark==0.9.1
conda==24.11.2
conda-libmamba-solver==24.11.1
conda-package-handling==2.4.0
conda-package-streaming==0.11.0
contourpy==1.3.1
crcmod==1.7
cromshell==2.0.0
cryptography==42.0.8
cupy-cuda12x==13.3.0
cwl-upgrader==1.2.12
cwl-utils==0.37
cwltool==3.1.20250110105449
cycler==0.12.1
Cython==3.0.11
daal==2025.4.0
dacite==1.8.1
db-dtypes==1.3.1
debugpy==1.8.11
decorator==4.4.2
defusedxml==0.7.1
Deprecated==1.2.15
dill==0.3.9
distlib==0.3.9
distro==1.9.0
dm-tree==0.1.8
docker==7.1.0
docopt==0.6.2
docstring-parser==0.16
dsub==0.5.0
entrypoints==0.4
etils==1.11.0
exceptiongroup==1.2.2
executing==2.1.0
explainable-ai-sdk==1.3.3
Farama-Notifications==0.0.4
fastapi==0.115.6
fastavro==1.10.0
fasteners==0.19
fastinterval==0.1.1
fastjsonschema==2.21.1
fastprogress==1.0.3
fastrlock==0.8.3
filelock==3.16.1
fiona==1.10.1
firecloud==0.16.38
flatbuffers==24.12.23
fonttools==4.55.3
fqdn==1.5.1
frozendict==2.4.6
frozenlist==1.5.0
fsspec==2024.12.0
funcsigs==1.0.2
gast==0.6.0
gatkpythonpackages==0.1
gcsfs==2024.12.0
geopandas==0.14.4
getm==0.0.5
ggplot==0.11.5
gitdb==4.0.11
GitPython==3.1.43
gnomad==0.6.4
google-api-core==2.19.0
google-api-python-client==2.131.0
google-apitools==0.5.31
google-auth==2.29.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==0.8.0
google-cloud-aiplatform==1.75.0
google-cloud-artifact-registry==1.14.0
google-cloud-batch==0.17.20
google-cloud-bigquery==2.34.4
google-cloud-bigquery-connection==1.17.0
google-cloud-bigquery-datatransfer==3.17.1
google-cloud-bigquery-storage==2.16.2
google-cloud-bigtable==1.7.3
google-cloud-core==2.4.1
google-cloud-datastore==1.15.5
google-cloud-dlp==3.26.0
google-cloud-functions==1.19.0
google-cloud-iam==2.17.0
google-cloud-jupyter-config==0.0.10
google-cloud-language==1.3.2
google-cloud-monitoring==2.24.0
google-cloud-pubsub==2.27.1
google-cloud-pubsublite==1.11.1
google-cloud-recommendations-ai==0.7.1
google-cloud-resource-manager==1.10.3
google-cloud-spanner==3.51.0
google-cloud-storage==1.44.0
google-cloud-videointelligence==1.16.3
google-cloud-vision==3.9.0
google-crc32c==1.6.0
google-pasta==0.2.0
google-resumable-media==2.7.2
googleapis-common-protos==1.66.0
gpustat==1.0.0
greenlet==3.1.1
grpc-google-iam-v1==0.12.7
grpc-interceptor==0.15.4
grpcio==1.68.1
grpcio-status==1.48.2
gs-chunked-io==0.5.2
gviz-api==1.10.0
gymnasium==1.0.0
h11==0.14.0
h2==4.1.0
h5netcdf==1.6.1
h5py==3.12.1
hail==0.2.134
hdbscan==0.8.40
hdfs==2.7.3
hpack==4.0.0
html5lib==1.1
htmlmin==0.1.12
httplib2==0.21.0
httptools==0.6.4
humanfriendly==10.0
humanize==4.11.0
hyperframe==6.0.1
ibis-framework==7.1.0
idna==3.10
igv-jupyter==1.0.0
ImageHash==4.3.1
imageio==2.36.1
immutabledict==4.2.1
importlib-metadata==8.4.0
importlib-resources==6.4.5
intel-cmplr-lib-ur==2025.1.0
intel-openmp==2025.1.0
ipykernel==6.29.5
ipython==8.21.0
ipython-genutils==0.2.0
ipython-sql==0.5.0
ipywidgets==8.1.5
isal==1.7.2
isodate==0.7.2
isoduration==20.11.0
isort==6.0.1
janus==1.0.0
jaraco.classes==3.4.0
jaraco.context==6.0.1
jaraco.functools==4.1.0
jedi==0.19.2
jeepney==0.8.0
Jinja2==3.0.3
jmespath==0.10.0
joblib==1.4.2
jproperties==2.1.2
json5==0.10.0
jsonpatch==1.33
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter-client==7.4.9
jupyter-contrib-core==0.4.2
jupyter-contrib-nbextensions==0.7.0
jupyter-core==5.3.1
jupyter-events==0.7.0
jupyter-highlight-selected-word==0.2.0
jupyter-http-over-ws==0.0.8
jupyter-nbextensions-configurator==0.6.3
jupyter-server==1.24.0
jupyter-server-fileid==0.9.3
jupyter-server-mathjax==0.2.6
jupyter-server-proxy==4.0.0
jupyter-server-terminals==0.4.4
jupyter-server-ydoc==0.8.0
jupyter-ydoc==0.2.5
jupyterlab==3.4.8
jupyterlab-git==0.42.0
jupyterlab-pygments==0.2.2
jupyterlab-server==2.23.0
jupyterlab-widgets==3.0.14
jupytext==1.15.0
keras==3.5.0
keras-tuner==1.4.7
kernels-mixer==0.0.15
keyring==25.5.0
keyrings.google-artifactregistry-auth==1.1.2
kfp==2.5.0
kfp-pipeline-spec==0.2.2
kfp-server-api==2.0.5
kiwisolver==1.4.8
kt-legacy==1.0.5
kubernetes==26.1.0
lazy-loader==0.4
libclang==18.1.1
libmambapy==2.0.5
linkify-it-py==2.0.3
llvmlite==0.41.1
lxml==5.3.2
lz4==4.3.3
mako==1.3.10
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.7.3
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
mccabe==0.7.0
mdit-py-plugins==0.4.2
mdurl==0.1.2
memray==1.15.0
menuinst==2.2.0
mistune==3.0.2
mizani==0.9.2
mkl==2025.1.0
ml-dtypes==0.4.1
mock==5.1.0
more-itertools==10.5.0
msal==1.32.0
msal-extensions==1.3.1
msgpack==1.1.0
msrest==0.7.1
multidict==6.1.0
multimethod==1.12
multipledispatch==1.0.0
multiprocess==0.70.18
mypy-extensions==1.0.0
namex==0.0.8
narwhals==1.35.0
nbclassic==0.4.8
nbclient==0.10.2
nbconvert==7.11.0
nbdime==3.2.0
nbformat==5.10.4
nbstripout==0.8.1
nest-asyncio==1.6.0
networkx==3.4.2
nose==1.3.7
notebook==6.5.4
notebook-executor==0.2
notebook-shim==0.2.3
numba==0.58.1
numpy==1.24.4
nvidia-cublas-cu12==12.3.4.1
nvidia-cuda-cupti-cu12==12.3.101
nvidia-cuda-nvcc-cu12==12.3.107
nvidia-cuda-nvrtc-cu12==12.3.107
nvidia-cuda-runtime-cu12==12.3.101
nvidia-cudnn-cu12==8.9.7.29
nvidia-cufft-cu12==11.0.12.1
nvidia-curand-cu12==10.3.4.107
nvidia-cusolver-cu12==11.5.4.101
nvidia-cusparse-cu12==12.2.0.103
nvidia-ml-py==11.495.46
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.3.101
oauth2client==4.1.3
oauthlib==3.2.2
objsize==0.6.1
opencensus==0.11.4
opencensus-context==0.1.3
opentelemetry-api==1.27.0
opentelemetry-exporter-otlp==1.27.0
opentelemetry-exporter-otlp-proto-common==1.27.0
opentelemetry-exporter-otlp-proto-grpc==1.27.0
opentelemetry-exporter-otlp-proto-http==1.27.0
opentelemetry-proto==1.27.0
opentelemetry-sdk==1.27.0
opentelemetry-semantic-conventions==0.48b0
opt-einsum==3.4.0
optree==0.13.1
orjson==3.10.12
overrides==7.7.0
packaging==21.3
pandas==2.0.3
pandas-gbq==0.17.9
pandas-profiling==3.6.6
pandocfilters==1.5.1
papermill==2.6.0
parameterized==0.9.0
parsimonious==0.10.0
parso==0.8.4
parsy==2.1
patsy==1.0.1
pdoc3==0.11.6
pendulum==3.0.0
pexpect==4.9.0
phik==0.12.4
pillow==11.0.0
pins==0.8.7
pip==25.1
platformdirs==4.3.6
plotly==5.24.1
plotnine==0.10.1
pluggy==1.5.0
prettytable==3.12.0
prometheus-client==0.21.1
promise==2.3
prompt-toolkit==3.0.48
propcache==0.2.1
proto-plus==1.25.0
protobuf==3.20.2
prov==1.5.1
psutil==5.9.3
ptyprocess==0.7.0
pulp==3.1.1
pure-eval==0.2.3
py-spy==0.4.0
py4j==0.10.9.9
pyarrow==9.0.0
pyarrow-hotfix==0.6
pyasn1==0.6.0
pyasn1-modules==0.4.0
pycares==4.6.1
pycosat==0.6.6
pycparser==2.22
pydantic==1.10.19
pydata-google-auth==1.9.0
pydot==1.4.2
pyfaidx==0.8.1.3
pyfasta==0.5.2
pygments==2.18.0
PyJWT==2.10.1
pylint==3.3.6
pymc3==3.11.4
pymongo==3.13.0
pyogrio==0.10.0
pyOpenSSL==24.0.0
pypandoc==1.15
pyparsing==3.2.0
pyproj==3.7.0
pysam==0.23.0
PySocks==1.7.1
python-datauri==3.0.2
python-dateutil==2.8.2
python-dotenv==1.0.1
python-json-logger==2.0.7
python-lzo==1.15
pytz==2023.3
PyVCF3==1.0.3
pywavelets==1.8.0
PyYAML==6.0.1
pyzmq==26.2.0
ray==2.40.0
rdflib==7.1.4
referencing==0.35.1
regex==2024.11.6
requests==2.32.3
requests-oauthlib==2.0.0
requests-toolbelt==0.10.1
retrying==1.3.4
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==12.6.0
rich-argparse==1.7.0
rpds-py==0.22.3
rsa==4.9
ruamel.yaml==0.18.6
ruamel.yaml.clib==0.2.8
s3transfer==0.12.0
schema-salad==8.9.20250408123006
scikit-image==0.25.0
scikit-learn==1.6.0
scikit-learn-intelex==2025.4.0
scipy==1.11.4
seaborn==0.12.2
SecretStorage==3.3.3
semver==3.0.4
Send2Trash==1.8.3
setuptools==70.3.0
Shapely==1.8.5.post1
shellingham==1.5.4
simpervisor==1.0.0
simple-parsing==0.1.6
six==1.17.0
slackclient==2.5.0
smart-open==7.1.0
smmap==5.0.1
sniffio==1.3.1
sortedcontainers==2.4.0
soupsieve==2.6
spython==0.3.14
SQLAlchemy==2.0.36
sqlglot==19.9.0
sqlparse==0.5.3
stack-data==0.6.3
starlette==0.41.3
statsmodels==0.14.4
tabulate==0.9.0
tangled-up-in-unicode==0.2.0
tbb==2022.1.0
tcmlib==1.3.0
tenacity==8.2.3
tensorboard==2.17.1
tensorboard-data-server==0.7.2
tensorboard-plugin-profile==2.18.0
tensorboardX==2.6.2.2
tensorflow==2.17.0
tensorflow-cloud==0.1.16
tensorflow-datasets==4.9.7
tensorflow-estimator==2.15.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==0.14.0
tensorflow-probability==0.25.0
tensorflow-serving-api==2.17.0
tensorflow-transform==0.14.0
termcolor==2.5.0
terminado==0.18.1
terra-notebook-utils==0.13.0
terra-widgets==0.0.1
textual==1.0.0
tf-keras==2.17.0
Theano==1.0.5
Theano-PyMC==1.1.2
threadpoolctl==3.5.0
tifffile==2024.12.12
time-machine==2.16.0
tinycss2==1.4.0
toml==0.10.2
tomli==2.2.1
tomlkit==0.13.2
toolz==0.12.1
tornado==6.4.2
tqdm==4.67.1
traitlets==5.9.0
truststore==0.10.0
typeguard==4.4.1
typer==0.15.1
types-python-dateutil==2.9.0.20241206
typing-extensions==4.12.2
tzdata==2024.2
uc-micro-py==1.0.3
umf==0.10.0
uri-template==1.3.0
uritemplate==3.0.1
urllib3==1.26.20
uvicorn==0.34.0
uvloop==0.21.0
virtualenv==20.28.0
visions==0.7.5
watchfiles==1.0.3
wcwidth==0.2.13
webcolors==24.11.1
webencodings==0.5.1
websocket-client==1.8.0
websockets==14.1
werkzeug==3.1.3
whatshap==2.7.dev2+g46b1b1d
wheel==0.45.1
widgetsnbextension==4.0.13
witwidget==1.8.1
wordcloud==1.9.4
wrapt==1.17.0
xarray==2023.12.0
xarray-einstats==0.8.0
xgboost==3.0.0
xopen==2.0.2
xxhash==3.5.0
xyzservices==2025.1.0
y-py==0.6.2
yarl==1.18.3
ydata-profiling==4.6.0
ypy-websocket==0.8.4
zipp==3.21.0
zlib-ng==0.5.1
zstandard==0.23.0

Relevant files:
- requirements.txt (for additional dependencies)
- configuration_cox.yaml (for config)
- model_cox.py (modeling)
- preprocessing_cox.py (preprocessing)
- dataloader_cox.py (data loading)


GOALS:

1) reduce additional dependencies - especially on mlflow, but also any dependencies that are pinned in requirements.txt that are already covered in the base AoU image. We can list them as dependencies (if they
are needed by the python files listed above), but attempt to avoid pinning to minimize conflicts

2) simplify overall codebase for ease of extension later

3) simplify API

For now, I'd like to do something like this:

```
!pip install git+https://<this repo>
```

I'd like to load the config from a dictionary read from a yaml, so that I can optionally define the config in pure python:

```
import yaml
from twinsight_model import CoxModelWrapper

config_dict = .... # load 'configuration_cox.yaml'

model = CoxModelWrapper(config = config_dict)
```

I'd like to be able to pretty simply load the data:

```
model.load_data()
model.train_data_summary() # info about # of patients, demographic breakdown, counts of outcomes and features...
```

Train it:

```
model.train(split = 0.8)
model.get_train_stats() # auroc, auprc, generate plots?
```

And use it:

```
pt_data = [{"age": 32, "bmi": 28.1, "heart_failure": True, ...}, {"age": 56, ...}]
predictions = model.get_prediction(pt_data) # 5 year survival probability, hazard ratio
```

Eventually I will want to export it to a .pkl file:

```
model.save_pickle('model.pkl')
```

And then later load it from pickle in another environment (assuming the deps are there), and use it to get predictions and other information:

```
model_rehydrated = .... # load from pickle

pt_data = [{"age": 32, "bmi": 28.1, "heart_failure": True, ...}, {"age": 56, ...}]
predictions = model_rehydrated.get_prediction(pt_data) # 5 year survival probability, hazard ratio
```

I also want to be able to get some of the stats as before:

```
model_rehydrated.train_data_summary()
model_rehydrated.get_train_stats()
```

HOWEVER, and this is IMPORTANT: no row-level data should be stored, and any count information that is stored should replace counts between 1 and 20 (inclusive) with "<20", since I am not allowed export any counts or 
other statistics based on patient groups of <20 (0 is ok).

Ideally, there would be some way for it to also tell me the required input types:

```
model_rehydrated.get_input_schema()
```

At this point, I don't really want to adjust the configuration schema that we've come up with, but if there's a way to use that info to inform the train_data_summary, train_stats, or input_schema, that could be good.



## suggested by AI:

# Simplified, production-ready API
from twinsight_model import CoxModelWrapper

# Initialize
model = CoxModelWrapper.from_pickle('copd_model.pkl')

# Basic prediction - validates input types and returns informative error
risk = model.predict_risk_batch([{"age": 65, "smoking": "current", "bmi": 28}])
# Returns: [{"1_year_risk": 0.05, "5_year_risk": 0.22, "hazard_ratio": 2.1}]

# Model information
info = model.get_model_info()
schema = model.get_input_schema()
importance = model.get_feature_importance()