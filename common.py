import os
import time
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset , DataLoader
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback

def wandb_login_init(project, name):
    load_dotenv()
    api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=api_key)

    train_time = datetime.fromtimestamp(time.time(), tz=ZoneInfo("Asia/Seoul")).strftime("%d-%H%M%S")
    wandb.init(project=project, name=f"{name}-{train_time}")
    
    print(f'name = {name}-{train_time}')


# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    def __init__(self,
            bos_token: str,
            eos_token: str,
        ) -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            return encoder_input.tolist()
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # Ground truth를 디코더의 input으로 사용하여 학습합니다.
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()


# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        
        # 번역 및 요약 훈련의 경우, decoder_input_ids를 제공해야 함.
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len


# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        
        # 번역 및 요약 훈련의 경우, decoder_input_ids를 제공해야 함.
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len


# Test에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len


# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'train.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(encoder_input_train,
                                         return_tensors = "pt",
                                         padding = True,
                                         add_special_tokens = True, 
                                         truncation = True, 
                                         max_length = config['tokenizer']['encoder_max_len'], 
                                         return_token_type_ids = False)
    
    tokenized_decoder_inputs = tokenizer(decoder_input_train, 
                                         return_tensors = "pt", 
                                         padding = True,
                                         add_special_tokens = True, 
                                         truncation = True, 
                                         max_length = config['tokenizer']['decoder_max_len'], 
                                         return_token_type_ids = False)
    
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, 
                                         return_tensors = "pt", 
                                         padding = True,
                                         add_special_tokens = True, 
                                         truncation = True, 
                                         max_length = config['tokenizer']['decoder_max_len'], 
                                         return_token_type_ids = False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs, len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, 
                                             return_tensors = "pt", 
                                             padding = True,
                                             add_special_tokens = True, 
                                             truncation = True, 
                                             max_length = config['tokenizer']['encoder_max_len'], 
                                             return_token_type_ids = False)
    
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, 
                                             return_tensors = "pt", 
                                             padding = True,
                                             add_special_tokens = True, 
                                             truncation = True, 
                                             max_length = config['tokenizer']['decoder_max_len'], 
                                             return_token_type_ids=False)
    
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, 
                                             return_tensors = "pt", 
                                             padding = True,
                                             add_special_tokens = True, 
                                             truncation = True, 
                                             max_length = config['tokenizer']['decoder_max_len'], 
                                             return_token_type_ids = False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs, len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset


# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
    print(f"results = {results}")

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {key: value["f"] for key, value in results.items()}
    return result


# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    # set training args
    training_args = Seq2SeqTrainingArguments(
                output_dir = config['general']['output_dir'], # model output directory
                overwrite_output_dir = config['training']['overwrite_output_dir'],
                num_train_epochs = config['training']['num_train_epochs'],  # total number of training epochs
                learning_rate = config['training']['learning_rate'], # learning_rate
                per_device_train_batch_size = config['training']['per_device_train_batch_size'], # batch size per device during training
                per_device_eval_batch_size = config['training']['per_device_eval_batch_size'],# batch size for evaluation
                warmup_ratio = config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
                weight_decay = config['training']['weight_decay'],  # strength of weight decay
                lr_scheduler_type = config['training']['lr_scheduler_type'],
                optim = config['training']['optim'],
                gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
                evaluation_strategy = config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
                save_strategy = config['training']['save_strategy'],
                save_total_limit = config['training']['save_total_limit'], # number of total save model.
                fp16 = config['training']['fp16'],
                load_best_model_at_end = config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
                seed = config['training']['seed'],
                logging_dir = config['training']['logging_dir'], # directory for storing logs
                logging_strategy = config['training']['logging_strategy'],
                predict_with_generate = config['training']['predict_with_generate'], #To use BLEU or ROUGE score
                generation_max_length = config['training']['generation_max_length'],
                do_train = config['training']['do_train'],
                do_eval = config['training']['do_eval'],
                report_to = config['training']['report_to'] # (선택) wandb를 사용할 때 설정합니다.
            )

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience = config['training']['early_stopping_patience'],
        early_stopping_threshold = config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model = generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args = training_args,
        train_dataset = train_inputs_dataset,
        eval_dataset = val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks = [MyCallback]
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer


# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer


# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, 
                                              return_tensors = "pt", 
                                              padding = True,
                                              add_special_tokens = True, 
                                              truncation = True, 
                                              max_length = config['tokenizer']['encoder_max_len'], 
                                              return_token_type_ids = False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset


# 추론을 위한 tokenizer와 학습시킨 모델을 불러옵니다.
def load_tokenizer_and_model_for_test(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model , tokenizer


# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model, tokenizer = load_tokenizer_and_model_for_test(config,device)

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids = item['input_ids'].to('cuda:0'), 
                                                    no_repeat_ngram_size = config['inference']['no_repeat_ngram_size'],
                                                    early_stopping = config['inference']['early_stopping'],
                                                    max_length = config['inference']['generate_max_length'],
                                                    num_beams = config['inference']['num_beams'],
                                                    )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output
