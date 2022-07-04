import torch
import transformers
from modeling_scaling_test import QILBertForSequenceClassification
from configuration_qilbert import QILBertConfig
from datasets import load_dataset, load_metric
import numpy as np
from ray import tune
import random
import time
from dataclasses import field, dataclass, asdict
from transformers import Trainer, TrainingArguments, AutoConfig, GlueDataTrainingArguments, AutoTokenizer, TrainerCallback
from quant_modules import QuantEmbedding
import torch.optim as opt
from torch.utils.data import DataLoader
import argparse
torch.autograd.set_detect_anomaly(True)
import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from ray.tune.suggest.optuna import OptunaSearch
import optuna
import wandb

class MyCallback(TrainerCallback):
    # A callback to wandb for center / distance parameter

    def on_evaluate(self, args, state, control, model = None, **kwargs):
        import wandb;
        param = {}
        target = ('c_W', 'd_W', 'c_A', 'd_A', 'gamma')
        if model is not None:
            for name, p in model.named_parameters():
                if name.endswith(target):
                    param[name] = {}
                    if len(p.shape) > 1:
                        for idx, data in enumerate(p.reshape(-1).data):
                            param[name]['head'+str(idx)] = data
                    else:
                        param[name]= p.data
        wandb.log(param)
                        
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha = 0.5, temperature =2.0, beta = 0.1, gamma = 1,**kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs = False):
        outputs_student = model(**inputs, output_attentions = True, output_hidden_states = True)
        # label loss
        student_loss = outputs_student.loss
        
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs, output_attentions = True, output_hidden_states = True)
        ## assert_size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()
        
        loss_function = nn.KLDivLoss(reduction = "batchmean")
        # student - teacher logits loss
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim = -1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim = -1)) * (self.args.temperature ** 2))
        
        attention_hidden_loss_function = nn.MSELoss()
        loss_attentions = 0.0
        loss_hiddens = 0.0
        for (attention_student, attention_teacher) in zip(outputs_student.attentions,outputs_teacher.attentions):
            loss_attentions += attention_hidden_loss_function(attention_student, attention_teacher) / attention_student.shape[1]

        for (hidden_student, hidden_teacher) in zip(outputs_student.hidden_states, outputs_teacher.hidden_states):
            loss_hiddens += attention_hidden_loss_function(hidden_student, hidden_teacher)

        # beta : hidden, gamma : attention
        loss = self.args.alpha * student_loss + ( 1. - self.args.alpha) * loss_logits + self.args.beta * loss_hiddens + self.args.gamma * loss_attentions
        return (loss, outputs_student) if return_outputs else loss

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_mode', action = 'store_true')
    parser.add_argument('--init_param', action = 'store_true')
    parser.add_argument('--after_search_finetuning', action = 'store_true')
    parser.add_argument('--debug_mode', action = 'store_true')
    parser.add_argument('--not_embedding_quant', action = 'store_true')
    parser.add_argument('--directory')
    parser.add_argument('--weight_plot', action = 'store_true')
    parser.add_argument('--print_percentage', action = 'store_true')
    parser.add_argument('--weight_bit', '-w', default = 8, type = int)
    parser.add_argument('--act_bit', '-a', default = 8, type = int)
    parser.add_argument('--distillation', action = 'store_true')
    parser.add_argument('--teacher_directory', '-t', default = None)
    parser.add_argument('--weight_init_mode', default = "min_max")
    parser.add_argument('--act_init_mode', default = "min_max")
    parser.add_argument('--box_plot', action = 'store_true')
    parser.add_argument('--epochs', default = 12, type = int)
    parser.add_argument('--group_wise', action = 'store_true')
    parser.add_argument('--num_groups', default = 12, type = int)
    parser.add_argument('--embedding_bit','-e', default = 8, type = int)
    args = parser.parse_args()
    return args

def set_random_seed(seed = 30):
    print('seed for random sampling : {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def experiment_setup():
    return './test' + str(time.time())

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    return tokenizer

def get_dataset():
    dataset = load_dataset('glue', 'cola')
    return dataset

def get_metric():
    metric_acc = load_metric('accuracy')
    metric_matthew = load_metric('glue', 'cola')

    return metric_acc, metric_matthew

def encode(examples):
    outputs = tokenizer(examples['sentence'], truncation = True, padding = True)
    return outputs

def get_teacher_model():
    model_path = os.path.join(args.teacher_directory, "pytorch_model.bin")
    check_point = torch.load(model_path)
    bert_config = QILBertConfig(group_wise = True)
    teacher_model = QILBertForSequenceClassification(bert_config)
    teacher_model.load_state_dict(check_point)
    return teacher_model

def get_model(config):

    bert_config = QILBertConfig(weight_bit = args.weight_bit, act_bit = args.act_bit, embedding_bit = args.embedding_bit, group_wise = args.group_wise, num_groups = args.num_groups,
                                quant_mode = args.quant_mode)
    if args.quant_mode:
        model = QILBertForSequenceClassification(bert_config)
        check_point = torch.load(os.path.join(args.directory, 'pytorch_model.bin'))
        if args.init_param:
            for key, value in check_point.copy().items():
                if "c_W" in key or "c_A" in key or "d_W" in key or "d_A" in key:
                    check_point.pop(key)
            model.load_state_dict(check_point, strict = False)
            model.eval()
            model.to(device)
            initialize_parameter(valid_dataloader, model)
            model.train()
        if args.after_search_finetuning:
            check_point = torch.load(os.path.join(args.directory, 'pytorch_model.bin'))
            model = QILBertForSequenceClassification(bert_config)
            for key, value in check_point.copy().items():
                if "c_A" in key or "d_A" in key or "alpha" in key or "c_W" in key or "d_W" in key or "c_bias" in key or "d_bias" in key or "beta" in key or "scaling_factor" in key or "offset" in key:
                    print(key,value)
                    check_point[key] = torch.Tensor([value])
                    print(check_point[key])
            model.load_state_dict(check_point, strict = False)
            for name, p in model.named_parameters():
                if "c_A" in name  or "d_A" in name or "gamma" in name or "c_W" in name or "d_W" in name or "c_bias" in name or "d_bias" in name:
                    p.requires_grad = False
        else:
            model.load_state_dict(check_point, strict = False)
            for name, p in model.named_parameters():
                if "c_A" in name or "c_W" in name:
                    print(p)

    else :
        check_point = torch.load(os.path.join(args.directory ,'pytorch_model.bin'))
        for key, value in check_point.copy().items():
            if "c_W" in key or "c_A" in key or "d_W" in key or "d_A" in key:
                check_point.pop(key)
        model = QILBertForSequenceClassification(bert_config)
        model.load_state_dict(check_point, strict = False) 
    if args.debug_mode:
        for name, m in model.named_modules():
            if hasattr(m, "debug_mode"):
                m.debug_mode = True
    else:
        for name, m in model.named_modules():
            if hasattr(m, "debug_mode"):
                m.debug_mode = False


    return model

def initialize_parameter(valid_dataloader, model):
    for m in model.modules():
        if hasattr(m, "quantize_reset_parameter"):
            m.quantize_reset_parameter(args)
    model.eval()
    for m in model.modules():
        if getattr(m, 'act_module_init', False) or getattr(m, 'weight_module_init', False):
            m.momentum = 0.95
            m.act_init_mode = True
            m.init_mode = args.act_init_mode
    
    with torch.no_grad():
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(len(valid_dataloader)))
        for epoch in range(1):
            for batch in valid_dataloader:
                b = {}
                for k, v in batch.items():
                    if k == "label":
                        b["labels"] = v.to(device)
                    else:
                        b[k] = v.to(device)
                outputs = model(**b)
                progress_bar.update(1)

    for m in model.modules():
        if getattr(m, 'act_module_init', False) or getattr(m, 'weight_module_init', False):
            m.act_init_mode = False

    for name, p in model.named_parameters():
        if "c_W" in name or "c_A" in name or "d_W" in name or "d_A" in name:
            print(name, p )

def layer_norm_calibration(valid_dataloader, model):
    for m in model.modules():
        if getattr(m, "elementwise_affine", False):
            m.reset_parameters()
            m.training = True
    optim = opt.Adam(model.parameters(), lr = 1e-5)
    with torch.no_grad():
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(len(valid_dataloader)))
        for batch in valid_dataloader:
            b = {}
            for k, v in batch.items():
                if k == "label": 
                    b["labels"] = v.to(device)
                else:
                    b[k] = v.to(device)
            outputs = model(**b)
            print(outputs['loss'])
            progress_bar.update(1)
            loss.backward()



def get_param_space():
    if args.distillation:
        space = {
                "learning_rate" : tune.grid_search([3e-5, 3e-6, 5e-6, 1.5e-6]),
                "per_device_train_batch_size" : tune.grid_search([16]),
                "alpha" : tune.uniform(0.1, 0.9),
                "temperature": tune.randint(1, 5),
                "beta" : tune.uniform(0.01, 1),
                "gamma": tune.uniform(0.1, 1),
                }
    else :
        space = {
                "learning_rate" : tune.grid_search([1e-5, 2e-5, 5e-5,1e-4]),
                "per_device_train_batch_size" : tune.grid_search([16]),
                }
    return space

def experiment_setup():
    experiment = "distil" + str(args.distillation) + "weight_bit" + str(args.weight_bit) + "act_bit" + str(args.act_bit)
    return experiment

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(len(predictions))
    if args.distillation:
        predictions = predictions[0].argmax(axis = -1)
    else:
        predictions = predictions.argmax(axis = -1)
    result = metric_matthew.compute(predictions = predictions, references = labels)
    accuracy = metric_acc.compute(predictions = predictions, references = labels)
    result.update(accuracy)
    return result

def objective_metric(metrics):
    return metrics["eval_matthews_correlation"]

if __name__ == "__main__":
    set_random_seed()
    args = arg_parse()
    experiment = experiment_setup()

    tokenizer = get_tokenizer()
    metric_acc, metric_matthew = get_metric()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.deivce("cpu")
    
    space = get_param_space()
    dataset = get_dataset()
    encoded_dataset = dataset.map(encode, batched = True, batch_size = None)
    valid_dataloader = encoded_dataset["validation"].remove_columns(["sentence", "idx"])
    valid_dataloader.set_format("torch")
    valid_dataloader = DataLoader(valid_dataloader, shuffle = False, batch_size = 16)
    
    if args.distillation:
        training_args = DistillationTrainingArguments(
                experiment,
                per_device_train_batch_size = 16, 
                per_device_eval_batch_size = 16, 
                evaluation_strategy = "steps", 
                learning_rate = 5e-7,
                eval_steps = 100, 
                do_train = True, 
                do_eval = True, 
                weight_decay = 0.1, 
                adam_beta1 = 0.9, 
                adam_beta2 = 0.98, 
                adam_epsilon = 1e-06, 
                lr_scheduler_type = "linear", 
                warmup_ratio = 0.06, 
                logging_strategy = "steps", 
                logging_dir = "./test/" + str(time.time()) + "/log", 
                logging_steps=20, 
                save_strategy = "steps", 
                save_steps =100, 
                save_total_limit = 1, 
                load_best_model_at_end = True, 
                metric_for_best_model = "eval_matthews_correlation", 
                greater_is_better = True, 
                num_train_epochs = args.epochs,
                alpha = 0.5,
                temperature = 4.0,
                beta = 0.1,
                gamma = 1,
                )
        teacher_model = get_teacher_model()
        trainer = DistillationTrainer(
                    args = training_args,
                    tokenizer = tokenizer,
                    teacher_model = teacher_model,
                    train_dataset = encoded_dataset['train'],
                    eval_dataset = encoded_dataset['validation'],
                    model_init = get_model,
                    compute_metrics = compute_metrics
                    )
        trainer.add_callback(MyCallback)
        trainer.hyperparameter_search(
                    hp_space = lambda _ : space,
                    compute_objective = objective_metric,
                    n_trials = 20,
                    direction = "maximize",
                    backend = "ray",
                    resources_per_trial = {"cpu": 16, "gpu" : 2}
                    )

    elif args.debug_mode:
        device = torch.device("cuda")
        model = get_model(None).to("cuda")
        optimizer = opt.Adam(model.parameters(), lr = 1e-5)
        
        dataset = get_dataset()
        encoded_dataset = dataset.map(encode, batched = True, batch_size = None)
        encoded_dataset["train"] = encoded_dataset["train"].remove_columns(["sentence", "idx"])
        encoded_dataset.set_format("torch")
        
        train_dataloader = DataLoader(encoded_dataset['train'], shuffle = False, batch_size = 1)
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(len(train_dataloader)))
        model.eval()
        for epoch in range(1):
            for batch in train_dataloader:
                b = {}
                for k, v in batch.items():
                    if k == "label":
                        b["labels"] = v.to(device)
                    else:
                        b[k] = v.to(device)
                outputs = model(**b)
                outputs.loss.backward()
                print(outputs.loss)
                for name,p in model.named_parameters():
                    print(p.grad, name)
                    input()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

                progress_bar.update(1)
    elif args.weight_plot:
        model = get_model(None).to(device)
        for name, m in model.named_modules():
            if hasattr(m, "bitW"):
                m.weight_plot(name)
    elif args.print_percentage:
        model = get_model(None).to(device)
        for name, m in model.named_modules():
            if hasattr(m, "bitW"):
                size, pru, cli = m.print_percentage(name)
                print(name, size, pru, cli)
    elif args.box_plot:
        model = get_model(None).to("cpu")
        for name, m in model.named_modules():
            if hasattr(m, "box_plot"):
                m.box_plot(name)

    else:
        training_args = TrainingArguments(
                experiment,
                per_device_train_batch_size = 16, 
                per_device_eval_batch_size = 16, 
                evaluation_strategy = "steps", 
                learning_rate = 5e-7,
                eval_steps = 100, 
                do_train = True, 
                do_eval = True, 
                weight_decay = 0.1, 
                adam_beta1 = 0.9, 
                adam_beta2 = 0.98, 
                adam_epsilon = 1e-06, 
                lr_scheduler_type = "linear", 
                warmup_ratio = 0.06, 
                logging_strategy = "steps", 
                logging_dir = "./test/" + str(time.time()) + "/log", 
                logging_steps=20, 
                save_strategy = "steps", 
                save_steps =100, 
                save_total_limit = 1, 
                load_best_model_at_end = True, 
                metric_for_best_model = "eval_matthews_correlation", 
                greater_is_better = True, 
                num_train_epochs = args.epochs,
                )
        trainer = Trainer(
                    args = training_args,
                    tokenizer = tokenizer,
                    train_dataset = encoded_dataset['train'],
                    eval_dataset = encoded_dataset['validation'],
                    model_init = get_model,
                    compute_metrics = compute_metrics
                    )
        trainer.add_callback(MyCallback)
        trainer.hyperparameter_search(
                    hp_space = lambda _ : space,
                    compute_objective = objective_metric,
                    n_trials = 1,
                    direction = "maximize",
                    backend = "ray",
                    resources_per_trial = {"cpu" : 8, "gpu":1}
                    )
