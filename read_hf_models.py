from huggingface_hub import hf_hub_download
from huggingface_hub import ModelFilter, DatasetFilter
from huggingface_hub import ModelSearchArguments, DatasetSearchArguments
from huggingface_hub import list_models
from huggingface_hub import HfApi
model_args = ModelSearchArguments()
dataset_args = DatasetSearchArguments()
filt = ModelFilter(
    task=model_args.pipeline_tag.TextClassification,
    trained_dataset=dataset_args.dataset_name.glue,
    library=model_args.library.PyTorch
)
hf_api = HfApi()
models = hf_api.list_models(filter=filt)[0]
list_models(filter=filt)[0]
hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json",cache_dir="s:/hf")
