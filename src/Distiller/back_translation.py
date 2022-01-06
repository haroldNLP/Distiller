from transformers import MarianMTModel, MarianTokenizer, FSMTForConditionalGeneration, FSMTTokenizer

class BackTranslationAugmenter:
    def __init__(self, from_model_name, to_model_name, device='cpu'):
        self.from_tokenizer = MarianTokenizer.from_pretrained(from_model_name)
        self.from_model = MarianMTModel.from_pretrained(from_model_name)
        self.to_tokenizer = MarianTokenizer.from_pretrained(to_model_name)
        self.to_model = MarianMTModel.from_pretrained(to_model_name)
        self.to_model.eval()
        self.from_model.eval()
        self.device = device
        self.from_model.to(device)
        self.to_model.to(device)

    def translate(self, texts, model, tokenizer, language):
        """Translate texts into a target language"""
        # Format the text as expected by the model
        formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
        original_texts = [formatter_fn(txt) for txt in texts]

        # Tokenize (text to tokens)
        tokens = tokenizer.prepare_seq2seq_batch(original_texts, return_tensors='pt')
        # Decode (tokens to text)
        translated_texts = tokenizer.batch_decode(model.generate(**tokens.to(self.device)), skip_special_tokens=True)

        return translated_texts

    def augment(self, texts, language_src='en', language_dst='fr', num_thread=1):
        """Implements back translation"""
        # Translate from source to target language
        translated = self.translate(texts, self.from_model, self.from_tokenizer, language_dst)

        # Translate from target language back to source language
        back_translated = self.translate(translated, self.to_model, self.to_tokenizer, language_src)

        return back_translated