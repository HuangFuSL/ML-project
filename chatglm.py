import zhipuai
import torch

# your api key

chatglm_args = {
    'model': 'text_embedding'
}

def build_prompt():
    pass

def invoke_example():
    response = zhipuai.model_api.invoke(
        prompt=[{'role': 'user', 'content': prompt_en}],
        **chatglm_args
    )
    print(torch.tensor(response['data']['embedding']).shape)

invoke_example()