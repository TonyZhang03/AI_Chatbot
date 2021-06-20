#from os.path import dirname
from aiogram import Bot, Dispatcher, types
#from aiogram.types import ChatType
from aiogram.utils import executor
from torch import tensor, argmax
#from transformers import cached_path
import re
import Config
#import speech_recognition as sr
#import torch
from models import load_model
#import tarfile
#import os
#from google.oauth2 import service_account
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import translate_v3 as translate
import logging
log = logging.getLogger(__name__)
logger = logging.getLogger(__file__)

#mycredential = service_account.Credentials.from_service_account_file("C:\PythonProject\AI_chatbot\ai-chatbot-311809-3ec75d9638ee.json")


CMN: str = None
TOKENIZER = None
MODEL = None
MNAME = dict()
MNAME["qnapipeline"] = "question-answering"
MNAME["qnabert"] = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad" #"lewtun/distilbert-base-uncased-finetuned-squad-v1"
MNAME["chatgpt"] = "download_gpt_cache"  
MNAME["chatgpt2chinese"] = "uer/gpt2-distil-chinese-cluecorpussmall"
MNAME["smarthome"] = "distilbert-base-uncased-finetuned-sst-2-english"
MNAME["transen-ch"] = "Helsinki-NLP/opus-mt-en-zh"
QNACONTEXT = """Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its share and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet leading subsidiary and will continue to be the umbrella company for Alphabets Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet."""
USERCONTEXT = None
SETCON = False
SMHOME = False
CLANG = "English"   # "ko-KR" for Korean, "ja-JP" for Japanese, "de-DE" for German, "zh-CN (cmn-Hans-CN)" for Chinese
HISTORY = []
PERSON = ["i am an ai chatbot built with gpt model .",
            "i can chat with you in english .",
            "i love talking to human to improve my language skill .",
            "i am currently hosting this chat group , policy is: be polite ."]
    

HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

# https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models

TELETOKEN = Config.MYTOKEN
bot = Bot(token = TELETOKEN)
dp = Dispatcher(bot)
# loop = asyncio.get_event_loop()
HOMEAP = {"tv": {"control":False, "status":"off", }, "light": {"control":False, "status":"off"}, "fan":{"control":False, "status":"off"}, "aircon":{"control":False, "status":"off", "temp":"28"}}
LANGC = {"Chinese":"zh-CN", "English":"en-US", "German":"de-DE", "Japanese":"ja-JP", "Korean":"ko-KR"}  # "chinese":"cmn-Hans-CN"

def trans(textin, src, target):
    client = translate.TranslationServiceClient.from_service_account_json('ai-chatbot-311809.json')
    location = "global"
    project_id = "ai-chatbot-311809"
    parent = f"projects/{project_id}/locations/{location}"
    #glossary = client.glossary_path(project_id, "us-central1", glossary_id)  # The location of the glossary
    #glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary)
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [textin],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": src,
            "target_language_code": target,
            "model": f"projects/{project_id}/locations/global/models/general/nmt",
            #"glossary_config": glossary_config,
        }
    )

    return response.glossary_translations[0].translated_text
    #for translation in response.translations:
        #print("Translated text: {}".format(translation.translated_text))


def get_text(content):
    global SLCODE
    client = speech.SpeechClient.from_service_account_json('ai-chatbot-311809.json')
    audio = speech.RecognitionAudio(content=content)
    if CLANG=="English":
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            sample_rate_hertz=16000,
            language_code="en-US",
            speech_context=[{"phrases":["aircon", ]}],
            model="command_and_search",
            use_emhamced=True,
        )
    else:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            sample_rate_hertz=16000,
            language_code="en-US",
            speech_context=[{"phrases":["aircon", ]}],
            model="command_and_search",
            use_emhamced=True,
            alternative_language_codes=LANGC[CLANG],
        )
    response = client.recognize(config=config, audio=audio)
    text = response.results[0].alternatives[0].transcript
    confidence = response.results[0].alternatives[0].confidence
    return text, confidence

def get_voice(text):
    client = texttospeech.TextToSpeechClient.from_service_account_json('ai-chatbot-311809.json')
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C",  # en-US-Standard-E, en-US-Wavenet-C, en-US-Wavenet-E, 
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS,
        speaking_rate=0.9,  # default 1,  0.5 is half as fast, 2 is twice as fast
        volume_gain_db=1,  # default 0, -6 is half volume as default, 6 is twice volume as default
        sample_rate_hertz=16000,
        effects_profile_id=['handset-class-device'],
    )
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content

async def get_reply(message: types.Message, textin=None):
    global CMN, MODEL, TOKENIZER, QNACONTEXT, USERCONTEXT, HISTORY, PERSON, SETCON, SMHOME, HOMEAP
    myreturn = dict()
    if textin is None:
        textin = message.text
    #if SMHOME:   
    if SETCON:
        USERCONTEXT = textin
        myreturn["modeln"] = "if you already set the bot in QnA mode, go ahead asking question or you need to issue a /qna command. You may cancel user context by command: /context none"
        myreturn["reply"] = "your context is reserved for QnA task with first priority, "
        SETCON = False
        return myreturn
    if CMN == 'qnapipeline':
        if USERCONTEXT is None:
            usecon = QNACONTEXT
        else:
            usecon = USERCONTEXT
        model = MODEL
        reply = model(question = textin, context = usecon)
        myreturn["modeln"] = " [Powered by Transformers.pipeline]"
        myreturn["reply"] = reply["answer"]
    elif CMN == 'chatgpt':
        model = MODEL
        result, histo = model.interact_single(message=textin, history= HISTORY, personality=PERSON)
        HISTORY = histo if len(histo)<=5 else histo[-5:]  # max_history=2
        myreturn["reply"] = result
        myreturn["modeln"] = " [Powered by ConvAI]"
    elif CMN == 'chatgpt2chinese':
        model = MODEL
        result = model(textin, max_length=40, repetition_penalty=1.3, do_sample=True, top_k=10)
        rawtext = result[0]["generated_text"]
        textl = rawtext.split('。')
        myreturn["reply"] = textl[0]
        myreturn["modeln"] = " [Powered by Transformers.pipeline]"
    elif CMN == 'qnabert':
        if USERCONTEXT is None:
            usecon = QNACONTEXT
        else:
            usecon = USERCONTEXT
        model = MODEL
        tokenizer = TOKENIZER
        input_ids = tokenizer.encode(textin, usecon)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        start_scores, end_scores = model(tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=tensor([segment_ids]), return_dict = False)
        answer_start = argmax(start_scores)
        answer_end = argmax(end_scores)
        # # Combine the tokens in the answer and print it out.
        answer = " ".join(tokens[answer_start:answer_end+1])
        final_reply = ''
        for word in answer.split():        
            #If it's a subword token
            if word[0:2] == '##':
                final_reply += word[2:]
            else:
                final_reply += ' ' + word
        myreturn["reply"] = final_reply
        myreturn["modeln"] = " [Powered by DistilBERT model]"
    elif CMN == 'transen-ch':
        model = MODEL
        myreturn["reply"] = model(textin, max_length=60)[0]['translation_text']
        myreturn["modeln"] = " [Powered by AIbot]"
    else:
        myreturn["reply"] = textin
        myreturn["modeln"] = " [No model is selected, this is echo bot.]"
    return myreturn

@dp.message_handler(commands= ['chat', 'qna', 'smarthome'])
async def loadnlp_command(message:types.Message):
    global MODEL, CMN, TOKENIZER, MNAME, USERCONTEXT, QNACONTEXT, CLANG
    cml =  ["qna", "chat", "smarthome"] 
    rcm = message.text
    rcml = rcm.split('@', maxsplit=1)
    cm = "".join(re.findall(r"[a-zA-Z0-9]+", rcml[0]))
    if cm not in cml:
        reply = "sorry, you entered invalid command, try again." 
    else:
        await message.reply("Loading designate NLP model... be patient.")
        firstp, secondp = load_model(cm, MNAME[cm])
        if firstp is None:
            reply = "sorry, "+cm+" NLP model is not loaded."
        else:
            MODEL = firstp
            TOKENIZER = secondp 
            CMN = cm
            if cm == "qna":
                if USERCONTEXT is None:
                    currcon = QNACONTEXT
                else:
                    currcon = USERCONTEXT
                reply = "Q&A NLP model loaded, please ask relevant question to below context: \n" + currcon +"\n[current language: "+CLANG+"]"
            elif cm == "chat":
                reply = "selected "+cm+" NLP model loaded, enjoy chatting with bot in "+CLANG
            else:
                reply = "selected "+cm+" NLP model loaded, enjoy smart home bot in "+CLANG
    await message.reply(reply)


@dp.message_handler(commands= ['help', 'lang', 'context'])
async def command_handler(message:types.Message):
    global SETCON, USERCONTEXT, QNACONTEXT, CLANG
    cml = ["help", "lang", "context", "langchinese", "langjapanese", "langkorean", "langgerman", "contextdel"] 
    rcm = message.text
    rcml = rcm.split('@', maxsplit=1)
    cm = "".join(re.findall(r"[a-zA-Z0-9]+", rcml[0]))
    if cm not in cml:
        reply = "sorry, you entered invalid command, try again." 
    else:    
        if cm == 'help':
            reply = "Hi, I am an AI chatbot able to understand your voice or text in five languages(default in English).\nYou can issue a command to set chatting mode/task or set chatting language. e.g. /qna will set bot at QnA tsak.\nBelow list all valid commands:\n/qna (set bot to QnA task)\n/context (send in user context for QnA)\n/context del (delete user context and set default context)\n/chat (set bot in free chat mode)\n/smarthome (set bot in smart home mode)\n/lang chinese (set bot chatting in Chinese)\n/lang korean (set bot chatting in Korean)\n/lang japanese (set bot chatting in Japanese)\n/lang german (set bot chatting in German)\n/lang (set bot chatting in English)" 
        elif cm == 'context':
            SETCON = True
            reply = "Please send a English context paragraph in next message as user context(you may copy paste a paragraph from Wiki). Note: existing user context (if any) will be overwritten, user contaxt will be retained and used for QnA task until being deleted by command: /context del"
        elif cm == 'contextnone':
            SETCON = False
            USERCONTEXT = None
            reply = "you have deleted current user context, a defualt context will be used in QnA task."
        elif cm == "lang":
            CLANG = "English"
            reply = "have set bot chatting in English."
        elif cm == "langchinese":
            CLANG = "Chinese"
            reply = "have set bot chatting in Chinese."
        elif cm == "langjapanese":
            CLANG = "Japanese"
            reply = "have set bot chatting in Japanese."
        elif cm == "langkorean":
            CLANG = "Korean"
            reply = "have set bot chatting in Korean."
        elif cm == "langgerman":
            CLANG = "German"
            reply = "have set bot chatting in German."
        else:
            reply = "wrong command."
    await message.reply(reply)


async def setup_bot_commands(dispatcher: Dispatcher):
    bot_commands = [
        types.BotCommand(command="/help", description="Get info about the bot"),
        types.BotCommand(command="/qna", description="Set bot in QnA task"),
        types.BotCommand(command="/chat", description="Set bot in free chat mode"),
        types.BotCommand(command="/smarthome", description="Set bot in Smart Home mode")
    ]
    await bot.set_my_commands(bot_commands)


@dp.message_handler(content_types = ["voice"])
async def voicechat(message: types.Message):
    file_id = message.voice.file_id
    cont = await bot.download_file_by_id(file_id)
    v_text, confi = get_text(cont.getvalue())
    print(v_text)
    print(confi)
    if confi < 0.75:
        reply_t = "I beg your pardon?"
    else:
        reply_t = get_reply(v_text)
    reply_v = get_voice(reply_t)
    await bot.send_voice(message.chat.id, reply_v)
    

#@dp.message_handler(chat_type = ChatType.GROUP)
@dp.message_handler(get_reply)
async def chat(message: types.Message, modeln: str, reply: str):
    textin = message.text
    if CLANG == "English":
        if textin == "bye" or textin == "Bye":
            await message.answer("Bye, see you later")
            await close_bot()
        else:
            reply = get_reply(textin)
            await message.answer(reply + modeln)
    else:
        text_e = trans(textin, src, "en-US")
        reply = get_reply(text_e)
        reply_t = trans(reply, "en-US", src)
        await message.answer(reply_t)


async def close_bot():
    dp.stop_polling()
    await dp.wait_closed()
    await bot.session.close()
    log.warning('bot is closed')

"""
def download_model():
    global MNAME, HF_FINETUNED_MODEL
    dirName = MNAME["chatgpt"]
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " + dirName +  " Created ")
        logger.info("downloading gpt_personachat_cache.tar.gz into memory...")
        resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
        logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, dirName))
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(dirName)
            print("completed extraction.")
    else:    
        print("Directory " + dirName +  " already exists")


def create_glossary(
    project_id="YOUR_PROJECT_ID",
    input_uri="YOUR_INPUT_URI",    ## this is glossary file path in cloud storage
    glossary_id="YOUR_GLOSSARY_ID",
    timeout=180,
):
    
    client = translate.TranslationServiceClient()
    # Supported language codes: https://cloud.google.com/translate/docs/languages
    source_lang_code = "en-US"  # must be BCP-47 code
    target_lang_code = "ja"     # must be BCP-47 code
    location = "us-central1"  # The location of the glossary

    name = client.glossary_path(project_id, location, glossary_id)
    language_codes_set = translate.types.Glossary.LanguageCodesSet(
        language_codes=[source_lang_code, target_lang_code]    # a list of your targeted languages codes
    )
    gcs_source = translate.types.GcsSource(input_uri=input_uri)
    input_config = translate.types.GlossaryInputConfig(gcs_source=gcs_source)
    glossary = translate.types.Glossary(name=name, language_codes_set=language_codes_set, input_config=input_config)
    parent = f"projects/{project_id}/locations/{location}"
    # glossary is a custom dictionary Translation API uses to translate the domain-specific terminology.
    # https://googleapis.dev/python/translation/latest/translate_v3/types.html#google.cloud.translate_v3.types.Glossary
    operation = client.create_glossary(parent=parent, glossary=glossary) 
    result = operation.result(timeout)
    print("Created: {}".format(result.name))
    print("Input Uri: {}".format(result.input_config.gcs_source.input_uri))
"""


if __name__ == '__main__':
    #download_model()
    executor.start_polling(dp, skip_updates=True, on_startup=setup_bot_commands)


    


