
#import spacy
from transformers import BertTokenizerFast, AutoModelWithLMHead


def load_model(name, modelname):
    case = {
            'chatgpt2chinese':chatgpt2chinese,
            'qnapipeline':qnapipeline,
            'chatgpt':chatgpt,
            'qnabert':qnabert,
            'transen-ch':transench,
            'smarthome':smarthome
            }
    func = case.get(name, invalid_mn)
    return func(modelname)

def invalid_mn():
    return None, None

def qnapipeline(name):
    from transformers import pipeline
    model = pipeline(name)
    return model, None

def smarthome(name):
    from simpletransformers.classification import ClassificationModel
    model = ClassificationModel("distilbert", name, use_cuda=False)
    return model, None

def chatgpt(name):
    from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
    model_args = ConvAIArgs()
    model_args.max_length = 40
    model_args.min_length = 5
    model = ConvAIModel("gpt", name, args=model_args, use_cuda=False)
    return model, None

def chatgpt2chinese(name):
    from transformers import BertTokenizer, TFGPT2LMHeadModel
    from transformers import TextGenerationPipeline
    tokenizer = BertTokenizer.from_pretrained(name)
    model = TFGPT2LMHeadModel.from_pretrained(name)
    text_generator = TextGenerationPipeline(model, tokenizer)
    return text_generator, None

def transench(name):
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    trans_engine = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
    return trans_engine, None

def qnabert(name):
    from transformers import BertForQuestionAnswering
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertForQuestionAnswering.from_pretrained(name)
    return model, tokenizer
    
