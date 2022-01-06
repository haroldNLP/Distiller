from nlpaug.augmenter import char as nac
from nlpaug.augmenter import word as naw
from nlpaug import flow as naf
from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util import Action
import random
import numpy as np
from Distiller.back_translation import BackTranslationAugmenter
import json

class AutoAugmenter:
    def __init__(self, aug_args, threads):
        augmenter_table = {"contextual": naw.ContextualWordEmbsAug,
                           "random": naw.RandomWordAug,
                           "back_translation": BackTranslationAugmenter}
        self.augs = []
        self.aug_names = []
        self.threads = threads
        for i in aug_args:
            if i:
                name = i.pop("aug_type")
                print(f"Load Augmenter {name}")
                self.aug_names.append(name)
                self.augs.append(augmenter_table.get(name)(**i))
        # self.aug = augmenter_table.get(aug_type)(**aug_args)

    @classmethod
    def from_config(cls, aug_type):
        augmenter_config_path = f"{aug_type}_augmenter_config.json"
        with open(augmenter_config_path) as f:
            aug_args = json.load(f)
        aug_args["aug_type"] = aug_type
        return cls(aug_args)

    @classmethod
    def init_pipeline(cls, w=None, threads=1, aug_p=0.3):
        config_list = [{
          "aug_type": "contextual",
          "model_type": "distilbert",
          "top_p": 0.8,
          "aug_p": aug_p,
          "device": "cuda:3"
            },{
            "aug_type": "back_translation",
            "from_model_name": "Helsinki-NLP/opus-mt-en-ROMANCE",
            "to_model_name": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "device": "cuda:2"
        },{
        "aug_type": "random",
        "action": "swap",
        "aug_p": aug_p
    }]
        selected_list = []
        aug_args = []
        if w is not None:
            for i in w:
                aug_args.append(config_list[i])
        else:
            for i in range(3):
                while True:
                    aug_config = random.choice(config_list)
                    if aug_config['aug_type'] not in selected_list:
                        selected_list.append(aug_config['aug_type'])
                        break
                if random.random()>0.5:
                    aug_args.append(aug_config)
        return cls(aug_args, threads)


    def augment(self, data):
        # result = []
        for aug in self.augs:
            data = aug.augment(data, num_thread=self.threads)
        return data
        # return self.aug.augment(data)

    def __len__(self):
        return len(self.augs)
        # return 1