import torch
import torchaudio
import sounddevice as sd
from net import Convolutional_Speaker_Identification


def most_prob_voting(List):
    """
    soft voting for multiple prediction.
    :param List:
    """
    return list(sorted(List.items(), key=lambda x: x[1]))[-1][0]


def split_wev(speech):
    """
    split file each 3 sec and return the splits as list.
    :param speech:
    """
    file_list = []
    if len(speech[0]) > 48000:
        speech = speech[0][44:]  # delete 44 header samples
        splits = int(len(speech) / 47956)  # number of splits
        speech = speech[None, :]  # reshape
        for j in range(splits):
            file_list.append(speech[0][j * 47956:(j + 1) * 47956])
    else:
        # if the file is less than 3 sec...
        print("file to small")
    return file_list


def record(freq=48000, duration=3):
    """
    If you want to record your audio file use this method.
    :param freq:
    :param duration:
    """
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)
    print("Recording...")
    sd.wait()
    transform = torchaudio.transforms.Resample(freq, 16_000)

    return transform(torch.Tensor(recording).T)


def get_model_and_dict(lan):
    """
    this function return the model that trained for the requested 'lan'.
    :param lan:
    """
    if lan == "en":
        lan_dict = {0: 'us_en', 1: 'england_en', 2: 'canada_en'}
        model = Convolutional_Speaker_Identification()
        model.load_state_dict(torch.load("models1/models/" + lan + "stat0.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, lan_dict
    elif lan == "ca":
        lan_dict = {0: 'balearic_ca', 1: 'central_ca', 2: 'valencian_ca'}
        model = Convolutional_Speaker_Identification()
        model.load_state_dict(torch.load("models1/models/" + lan + "stat0.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, lan_dict
    elif lan == "fr":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'canada_fr', 1: 'france_fr', 2: 'belgium_fr', 3: 'france_fr'}
        model.eval()
        return model, lan_dict
    elif lan == "eu":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'mendebalekoa_eu', 1: 'erdialdekoa_nafarra_eu'}
        model.eval()
        return model, lan_dict
    elif lan == "zh-CN":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: '440000_zh-CN', 1: '450000_zh-CN', 2: '110000_zh-CN', 3: '330000_zh-CN'}
        model.eval()
        return model, lan_dict
    elif lan == "es":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'andino_es', 1: 'nortepeninsular_es', 2: 'chileno_es'}
        model.eval()
        return model, lan_dict
    elif lan == "de":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'switzerland_de', 1: 'austria_de', 2: 'germany_de'}
        model.eval()
        return model, lan_dict
    elif lan == "lan":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'eu', 1: 'de', 2: 'es', 3: 'ca', 4: 'fr', 5: 'zh-CN', 6: 'en'}
        model.eval()
        return model, lan_dict
    else:
        print(lan == "zh-CN")


def pedict_from_list(file_list, res_dict, model1):
    """
    Get list of 3 sec audio files (from split_wev), dict with the labels and their representing number
    and the relevant model.
    predict each 3 sec file and return finale answer.
    :param file_list:
    :param res_dict:
    :param model1:
    :return:
    """
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model()
    frequents = {}  # dict with accent and the sum of the model prediction probability for each accent.
    for f in file_list:
        f = f[None, :]  # reshape the file
        with torch.inference_mode():
            emission, _ = w2v(f)

        x = emission[None, :].clone()  # reshape the file
        softmax = torch.exp(model1(x))  # get softmax from log softmax
        out = torch.argmax(softmax)  # get the prediction from the softmax

        print(x.size())

        for i, x in enumerate(softmax[0]):
            print(f'{res_dict[i]} = {x * 100} %', )

        proba = int(torch.max(softmax) * 100)
        print("model 1 is " + str(proba) + " % sure")
        res = res_dict[int(out)]
        print(res)
        print("--------")
        # sum probabilities
        if res not in frequents.keys():
            frequents[res] = 0
        if res == "zh-CN":
            frequents[res] += proba * 0.3
        else:
            frequents[res] += proba

    return most_prob_voting(frequents)


def pred(file_name='fr_belgium.mp3'):
    speech_array, sampling_rate = torchaudio.load(file_name, normalize=True)
    # convert to 16000 sampling rate
    transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = transform(speech_array)

    # if you want to record by yourself...
    # speech_array = record()

    files = split_wev(speech_array)
    # predict language
    lan_model, result_dict = get_model_and_dict("lan")
    selected_lan = str(pedict_from_list(files, result_dict, lan_model))

    # print("\n The language model select: " + selected_lan)
    # print("\n--------")

    # predict accent
    accent_model, result_dict = get_model_and_dict(selected_lan)
    # print("\n |The answer is : " + pedict_from_list(files, result_dict, accent_model) + "|")
    return "\n The language model select: " + selected_lan + "\n--------" + "\n |The answer is : " + pedict_from_list(
        files, result_dict, accent_model) + "|"


print(pred())
