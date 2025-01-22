import json
import numpy as np
import random, math
import argparse,torch
import os
import json, tqdm, requests
import yaml
import time

class OpenAIAPIModel():
    # def __init__(self, api_key, url="https://api.openai.com/v1/chat/completions", model="gpt-4o-2024-11-20"):
    def __init__(self, api_key, url="https://api.openai.com/v1/chat/completions", model="gpt-4o-mini"):
        self.url = url
        self.model = model
        self.API_KEY = api_key

    def generate(self, text: str, temperature=0.7, system="You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.", top_p=1):
        headers={"Authorization": f"Bearer {self.API_KEY}"}

        query = {
            "model": self.model,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "stream": False
        }
        while True:
            responses = requests.post(self.url, headers=headers, json=query)
            print(responses)
            if 'choices' not in responses.json():
                print(text)
                print(responses)
                print(type(responses))
                print(responses.json())
                if responses.json()['error']['code'] == 429 or responses.json()['error']['code'] == 'rate_limit_exceeded':
                    print("Rate limit exceeded")
                    time.sleep(3)
                    continue
            break
            
        return responses.json()['choices'][0]['message']['content']
    
def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']
        

    docs = instance['doc']
    for i in range  (len(docs)):
        if len(docs[i]) >= 512:
            docs[i] = docs[i][:512]

    random.shuffle(docs)
    
    return query, docs, ans


def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False 
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def getevalue(results):
    results = np.array(results)
    results = np.max(results,axis = 0)
    if 0 in results:
        return False
    else:
        return True
            

def predict(query, ans, docs, model, system, instruction, temperature, dataset):

    '''
    label: 0 for positive, 1 for negative, -1 for not enough information

    '''
    if len(docs) == 0:

        text = instruction.format(QUERY=query, DOCS='')
        prediction = model.generate(text, temperature)

    else:

        docs = '\n'.join(docs)

        

        text = instruction.format(QUERY=query, DOCS=docs)

        prediction = model.generate(text, temperature, system)

        # print(prediction)
    # change a string with format to a dict
    result = json.loads(prediction)
    label = result['Label']
    prediction = result['Answer']
    # print(label, prediction)
    if 'insufficient information' in prediction:
        label = -1
    else:
        labels = checkanswer(prediction, ans)

    return label, prediction



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, default='knn',
        help='evaluetion dataset',
        choices=['knn', 'scann_95', 'gt', 'diskann_95', 'diskann_90', 'scann_90', 'non_gt']
    )
    parser.add_argument(
        '--api_key', type=str, default=os.environ.get("GPT_API", "None"),
        help='api key of chatgpt'
    )
    parser.add_argument(
        '--plm', type=str, default='THUDM/chatglm-6b',
        help='name of plm'
    )
    parser.add_argument(
        '--url', type=str, default='https://api.openai.com/v1/completions',
        help='url of chatgpt'
    )
    parser.add_argument(
        '--temp', type=float, default=0.2,
        help='corpus id'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.0,
        help='rate of noisy passages'
    )
    parser.add_argument(
        '--correct_rate', type=float, default=0.0,
        help='rate of correct passages'
    )
    parser.add_argument(
        '--passage_num', type=int, default=10,
        help='number of external passages'
    )
    parser.add_argument(
        '--factchecking', type=bool, default=False,
        help='whether to fact checking'
    )
    
    args = parser.parse_args()

    modelname = args.modelname
    temperature = args.temp
    noise_rate = args.noise_rate
    passage_num = args.passage_num

    instances = []
    with open(f'data/{args.dataset}.json','r') as f:
        for line in f:
            instances.append(json.loads(line))
    # instances = instances[:10]
    resultpath = f'result-{args.dataset}'
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    
    prompt = yaml.load(open('prompt.yaml', 'r'), Loader=yaml.FullLoader)['en']

    system = prompt['system']
    instruction = prompt['instruction']

    model = OpenAIAPIModel(api_key = args.api_key)
    


    filename = f'{resultpath}/prediction_{args.dataset}_{modelname}_temp{temperature}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}.json'
    useddata = {}
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data
    # print(useddata[3].keys())
    results = []
    skiped = 0
    with open(filename,'w') as f:
        for instance in tqdm.tqdm(instances):
            if instance['id'] in useddata and instance['query'] == useddata[instance['id']]['query'] and useddata[instance['id']]['label'] == 1:
                skiped = skiped + 1
                print("skip cached id:", instance['id'])

                results.append(useddata[instance['id']])
                f.write(json.dumps(useddata[instance['id']], ensure_ascii=False)+'\n')
                continue
            try:
                # print(instance)
                random.seed(2333)
                if passage_num == 0:
                    query = instance['query']
                    ans = instance['answer']
                    docs = []
                else:
                    query, docs, ans = processdata(instance, noise_rate, passage_num, args.dataset, args.correct_rate)
                labels, prediction,factlabel = predict(query, ans, docs, model, system,instruction,temperature,args.dataset)
                # instance['label'] = label
                newinstance = {
                    'id': instance['id'],
                    'query': query,
                    'docs': docs,
                    'label': labels,
                    'ans': ans,
                    'noise_rate': noise_rate,
                    'factlabel': factlabel
                }
                
                results.append(newinstance)
                f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')
            except Exception as e:
                print("Error:", e)
                continue
    tt = 0
    for i in results:
        label = i['label']
        if noise_rate == 1 and label == -1:
            tt += 1
        elif label == 1:
            tt += 1
    print(tt/len(results))
    scores = {
    'all_rate': (tt)/len(results),
    'noise_rate': noise_rate,
    'tt':tt,
    'nums': len(results),
    }
    if '_fact' in args.dataset:
        fact_tt = 0
        correct_tt = 0
        for i in results:
            if i['factlabel'] == 1:
                fact_tt += 1
                if 0 not in i['label']:
                    correct_tt += 1
        fact_check_rate = fact_tt/len(results)
        if fact_tt > 0:
            correct_rate = correct_tt/fact_tt
        else:
            correct_rate = 0
        scores['fact_check_rate'] = fact_check_rate
        scores['correct_rate'] = correct_rate
        scores['fact_tt'] = fact_tt
        scores['correct_tt'] = correct_tt

    

    json.dump(scores,open(f'{resultpath}/prediction_{args.dataset}_{modelname}_temp{temperature}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}_result.json','w'),ensure_ascii=False,indent=4)
