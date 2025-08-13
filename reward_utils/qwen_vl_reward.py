from openai import OpenAI
import base64
from io import BytesIO
import re
from json_repair import repair_json
import json

# Set OpenAI's API key and API base to use vLLM's API server.

prompt = """
You are given a text prompt used to generate image: "{}"
Below is one generated image: <image>
1. Describe the image thoroughly (objects, colors, layout,
etc.), do not be affected by the prompt.
2. Score the generation quality from following aspects:
- Is the object in the generated image satisfying the category according to the prompt? (0-1 score)
- Is the generated image completed without missing parts? (0-1 score)
- Does the content of generated image look real and reasonable? (0-1 score)
- Is the generated image clear, bright and in high quality? (0-1 score)
- Is the generated image free of any noises, defects and artifacts? (0-1 score)
"""
format_rules="""
Your response should in a **JSON** format following rules below:
1. The final response should have three keys: "description", "score", "explanation".
2. The total score of the image should be store in "score" key, you may use float numbers for the scores. The value of total score should be between 0 and 5.
3. The reasoning and scoring process could be write in "explanation" for you to explain your score.
Following is a exmaple of response:
```json
{
"description": "The image shows a brown dog with one eye and two ears. its mouth is open and the its tongue is sticking out.",
"score": 3.5,
"explanation": "The image shows a brown dog matches the prompt 'a photo of a dog' (1 score). The image is complete and all part can be seen (1 score). The dog in the photo looks unreal becaues it has only one eye and the eye position is not reasonable (0 score). The dog's face is clear but its body and backgroud is blurred (0.5 score). The image do not have any noises, defects and artifacts (1 score). So the total score of this image is 1+1+0+0.5+1=3.5 .'"
}
```
"""

def init_client(openai_api_key = "EMPTY", url="localhost", port="20004", interf="v1"):
    openai_api_base = f"http://{url}:{port}/{interf}"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def init_pipeline(model_or_path="Qwen/Qwen2.5-VL-3B-Instruct",device=None,**kwargs):
    from transformers import pipeline
    #print("pipeline on",device)
    pipe=pipeline("image-text-to-text",model=model_or_path,device_map=device,**kwargs)
    return pipe

def encode_base64_from_img(img):
    buffered = BytesIO()
    img.save(buffered,format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

default_model_name="Qwen2.5-VL-3B-Instruct"
defalut_messages = [
    {"role": "user", "content": None}
]

def question_image_score_with_lbl_vllm(img,lbl, client=None, model_name=default_model_name, gen_kwargs={"max_new_tokens":256}, messages=defalut_messages):
    img_s=encode_base64_from_img(img)
    prompt_txt=prompt.format(lbl)+format_rules
    if client is None:
        client=init_client()
    messages[0]["content"]=[{"type":"text","text":prompt_txt},
        {   
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_s}"}
        }
    ]
    out1 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 10,
        },
    )
    out1=out1.choices[0].message.content
    #print(out1)
    # assuming the output is format with json
    json_s_match=re.search(r"```json\n(.*?)\n```",out1,re.S)
    json_s_raw = json_s_match.group(1).strip() if json_s_match else out1.strip()
    json_s=repair_json(json_s_raw.strip())
    res=json.loads(json_s)
    #print(res)
    return res,client


def question_image_score_with_lbl_pipe(img,lbl, pipe=None, model_name=default_model_name, gen_kwargs={"max_new_tokens":256}, messages=defalut_messages):
    img_s=encode_base64_from_img(img)
    prompt_txt=prompt.format(lbl)+format_rules
    if pipe is None:
        pipe=init_pipeline()
    messages[0]["content"]=[{"type":"text","text":prompt_txt},
        {   
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_s}"}
        }
    ]
    with pipe.device_placement():
        out1=pipe(text=messages,**gen_kwargs)
    out1=out1[0]['generated_text'][-1]['content']
    #print(out1)
    # assuming the output is format with json
    json_s_match=re.search(r"```json\n(.*?)\n```",out1,re.S)
    json_s_raw = json_s_match.group(1).strip() if json_s_match else out1.strip()
    json_s=repair_json(json_s_raw.strip())
    res=json.loads(json_s)
    #print(res)
    return res,pipe

prompt_dict={
    "fake_judge":"""<image>
You need to inspect the image carefully and give a score according to the question below:
- Is the image an AI-generated image? If yes, give 0 score, otherwise give 1 score.
The answer should be in a JSON format like below:
```json
{
    "score": 0,
}
```
""",
    "weird_judge":"""<image>
You need to inspect the image carefully and give a score according to the question below:
- Is there any strange feature in the image? If yes, give 0 score, otherwise give 1 score.
The answer should be in a JSON format like below:
```json
{
    "score": 0,
}
```
""",
}

def question_image_score_prompt_key_pipe(img, prompt_key, pipe=None, model_name=default_model_name, gen_kwargs={"max_new_tokens":256}, messages=defalut_messages):
    img_s=encode_base64_from_img(img)
    prompt_txt=prompt_dict[prompt_key]
    if pipe is None:
        pipe=init_pipeline()
    messages[0]["content"]=[{"type":"text","text":prompt_txt},
        {   
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_s}"}
        }
    ]
    with pipe.device_placement():
        out1=pipe(text=messages,**gen_kwargs)
    out1=out1[0]['generated_text'][-1]['content']
    #print(out1)
    # assuming the output is format with json
    json_s_match=re.search(r"```json\n(.*?)\n```",out1,re.S)
    json_s_raw = json_s_match.group(1).strip() if json_s_match else out1.strip()
    json_s=repair_json(json_s_raw.strip())
    res=json.loads(json_s)
    #print(res)
    return res,pipe
