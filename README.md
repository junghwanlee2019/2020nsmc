# 2020nsmc

네이버영화평점분류자료 실행방법

1) 본인의 컴퓨터 혹은 구글 Colab 라이브러리를 DATA PATH로 잡아준다


from google.colab import drive 
drive.mount('/content/gdrive')

DATA_PATH = 'gdrive/My Drive/Colab Notebooks/'
import sys
sys.path.append(DATA_PATH)



2) 트랜스포머를 설치하고 필요한 각종 라이브러리를 설치해준다


!pip install transformers

import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime



3) 데이터를 다운로드한다. 판다스로 데이터를 불러올때 최종 테스트 데이터는 DATA_PATH를 설정해주어야 한다. 


!git clone https://github.com/e9t/nsmc.git

train = pd.read_csv("nsmc/ratings_train.txt", sep='\t')
test = pd.read_csv("nsmc/ratings_test.txt", sep='\t')
sec_test = pd.read_csv(DATA_PATH + "ko_data.csv", sep=',', encoding='CP949')

print(train.shape)
print(test.shape)
print(sec_test.shape)



4) 훈련 데이터셋에 리뷰 및 라벨데이터를 넣어주고 버트에 맞도록 형식 변경을 해준다. 버트는 토크나이저를 통하여 토큰화해준다. 
데이터의 길이를 설정해줄 수 있는데 영화 평점 데이터이므로 길이는 128이면 충분하다. 이후 토큰을 숫자 인덱스로 변환해준다.
테스트데이터셋도 같은 과정을 거친다. 


sentences = train['document']
sentences[:10]

sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
sentences[:10]

labels = train['label'].values
labels

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
MAX_LEN = 128

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids[0]


5) 어텐션 마스크를 설정해준다. 그리고 데이터와 어텐션 마스크를 트레이닝과 검증셋으로 분리해준다. 이후 파이토치로 수행하기 위해 파이토치의 텐서로 변경을 수행해준다. 


attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=2018, 
                                                                                    test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=2018, 
                                                       test_size=0.1)
                                                       
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)				

6) 적당한 배치사이즈를 설정해준다. 이는 추후 하이퍼 파라메타로 사용한다. 
   파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정

batch_size = 30

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

7) 같은 과정을 테스트데이터셋이도 반복해준다. 

sentences = test['document']
sentences[:10]

sec_sentences = sec_test['Sentence']
sec_sentences[:10]

sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
sec_sentence = ["[CLS] " + str(sec_sentence) + " [SEP]" for sec_sentence in sec_sentences]

sentences[:10]

labels = test['label'].values
labels

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
sec_tokenized_texts = [tokenizer.tokenize(sent) for sent in sec_sentence]

print (sentences[0])
print (tokenized_texts[0])
print (sec_tokenized_texts[0])

MAX_LEN = 128

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
sec_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in sec_tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
sec_input_ids = pad_sequences(sec_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids[0]

attention_masks = []
sec_attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

for seq in sec_input_ids:
    seq_mask = [float(i>0) for i in seq]
    sec_attention_masks.append(seq_mask)      

print(attention_masks[0])

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)
sec_test_id = torch.tensor(sec_test['Id'])
sec_test_inputs = torch.tensor(sec_input_ids)
sec_test_masks = torch.tensor(sec_attention_masks)

print(sec_test_id[0])
print(sec_test_inputs[0])
print(sec_test_masks[0])

batch_size = 30

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

sec_test_data = TensorDataset(sec_test_id, sec_test_inputs, sec_test_masks)
sec_test_dataloader = DataLoader(sec_test_data, batch_size=1)

8) GPU 사용설정하고 BERT 모델을 생성산다. 

device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

9) 옵티마이저 설정을 해준다.  학습률과 에포크수는 하이퍼파라메터가 될 수 있다. 

optimizer = AdamW(model.parameters(),
                  lr = 2e-6, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

epochs = 1

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
                                            
10) 정확도 계산 함수 등을 설정해준다. 

def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):

    # 반올림
    elapsed_rounded = int(round((elapsed)))
    
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model.zero_grad()

for epoch_i in range(0, epochs):

11) 트레이닝과 벨리데이션을 수행한다.
  
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_loss = 0
    
    model.train()
        
    for step, batch in enumerate(train_dataloader):

        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)
        
     
        b_input_ids, b_input_mask, b_labels = batch
              
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        loss = outputs[0]

     
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
       
        batch = tuple(t.to(device) for t in batch)
        
  
        b_input_ids, b_input_mask, b_labels = batch
        
     
        with torch.no_grad():     
          
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
      
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("training complete")

12) 테스트셋을 평가한다. 최종 CSV 파일로 평가한 자료가 내려받아진다. 최종 정확도가 산출된다. 

t0 = time.time()
batch = 1

model.eval()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
sec_result = []

for step, batch in enumerate(sec_test_dataloader):

    if step % 1000 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(sec_test_dataloader), elapsed))

    batch = tuple(t.to(device) for t in batch)
    
    b_id, b_input_ids, b_input_mask = batch
    
    with torch.no_grad():     
  
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
 
    logits = outputs[0]

    logits = logits.detach().cpu().numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()
    b_id = b_id.cpu().numpy()

    result = np.concatenate((b_id, pred_flat), axis=None)
        
    sec_result.append(result)    


rdf = pd.DataFrame(sec_result, columns =['Id', 'Predicted'])
rdf.to_csv(DATA_PATH + 'sample_original.csv', index=False)

sec_result[:10]


t0 = time.time()


model.eval()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0


for step, batch in enumerate(test_dataloader):
  
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch
   
    with torch.no_grad():     

        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))




