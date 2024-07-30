import os
import json
import torch
from transformers import BertTokenizer
from flask import Flask, request, jsonify

from stigma import settings
from stigma.util import chunks
from stigma import text_utils
from stigma.model import bert as bert_utils

app = Flask(__name__)

class StigmaSearch(object):
    def __init__(self, task2keyword=settings.CAT2KEYS, context_size=10):
        self._keyword_search = text_utils.KeywordSearch(task2keyword)
        self._context_size = context_size

    def __repr__(self):
        return f"StigmaSearch(context_size={self._context_size})"

    @staticmethod
    def show_default_keyword_categories():
        keycat_str = json.dumps(settings.CAT2KEYS, indent=1)
        print(keycat_str)

    def search(self, text):
        if isinstance(text, str):
            print(">> WARNING - Expected list of strings as input. Transforming str object to list.")
            text = [text]
        if not isinstance(text, list):
            raise ValueError("Text should be a list of strings.")
        matches = self._keyword_search.search_batch(text)
        matches_flat = []
        for doc_id, match in enumerate(matches):
            for match_dict in match:
                matches_flat.append({
                    "document_id": doc_id,
                    **match_dict,
                    "context": text_utils.get_context_window(
                        text=text[doc_id],
                        start=match_dict["start"],
                        end=match_dict["end"],
                        window_size=self._context_size,
                        clean_normalize=True,
                        strip_operators=True)
                })
        for mf in matches_flat:
            mf["keyword_category"] = mf.pop("task")
            mf["text"] = mf.pop("context")
        return matches_flat

class StigmaBertModel(object):
    def __init__(self, model, keyword_category, tokenizer=None, batch_size=16, device="cpu", **kwargs):
        if keyword_category not in settings.CAT2KEYS:
            print(">> WARNING: Received unexpected keyword_category not found in settings.CAT2KEYS")
        if device not in ["cpu", "cuda"]:
            raise ValueError("Expected 'device' to be one of 'cpu' or 'cuda'.")
        if batch_size <= 0:
            raise ValueError("Batch size should be at least 1.")
        self._device = device
        self._keyword_category = keyword_category
        self._model_name = model
        if model not in settings.MODELS:
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model not found ('{model}')")
            self._model = model
        else:
            if keyword_category not in settings.MODELS[model]["tasks"]:
                raise KeyError(f"Keyword category ({keyword_category}) not found in model directory.")
            if not os.path.exists(settings.MODELS[model]["tasks"][keyword_category]):
                raise FileNotFoundError("Model not found ('{}')".format(settings.MODELS[model]["tasks"][keyword_category]))
            if settings.MODELS[model]["model_type"] != "bert":
                raise ValueError("Specified a non-bert model. For baseline models, please use StigmaBaselineModel instead.")
            self._model = settings.MODELS[model]["tasks"][keyword_category]
        if tokenizer is None:
            tokenizer_path = r"C:\Users\u249391\Downloads\stigma\BioClinicalBERT"
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self._tokenizer = tokenizer
        _ = self._initialize_tokenizer(self._tokenizer)
        _ = self._initialize_model(self._model)
        _ = self.update_eval_batch_size(batch_size)

    def __repr__(self):
        return f"StigmaBertModel(model='{self._model_name}', keyword_category='{self._keyword_category}')"

    @staticmethod
    def show_default_models():
        models_str = json.dumps({x: y for x, y in settings.MODELS.items() if y.get("model_type", None) == "bert"}, indent=1)
        print(models_str)

    def update_eval_batch_size(self, batch_size):
        self._batch_size = batch_size

    def _initialize_tokenizer(self, tokenizer):
        tokenizer_path = r"C:\Users\u249391\Downloads\stigma\data\resources\models\BioClinicalBERT"
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def _initialize_model(self, model):
        if not os.path.exists(f"{model}/init.pth"):
            raise FileNotFoundError(f"Unable to find expected model init file: {model}/init.pth")
        if not os.path.exists(f"{model}/model.pth"):
            raise FileNotFoundError(f"Unable to find expected model weights file: {model}/model.pth")
        print("[Loading Model Parameters]")
        init_param_file = f"{model}/init.pth"
        init_params = torch.load(init_param_file)
        task_id = {y: x for x, y in enumerate(init_params["task_targets"])}
        if self._keyword_category not in task_id:
            raise KeyError(f"Model keyword_category ('{self._keyword_category}') not found in specified model task set ({task_id})")
        task_id = task_id[self._keyword_category]
        targets = sorted(init_params["task_targets"][self._keyword_category], key=lambda x: init_params["task_targets"][self._keyword_category][x])
        self.model_dict = {
            "task_id": task_id,
            "targets": targets,
            "init_params": init_params,
        }
        if init_params["classifier_only"]:
            print("[Initializing Encoder]")
            self.model_dict["encoder"] = bert_utils.BERTEncoder(
                checkpoint=init_params["checkpoint"],
                pool=True,
                use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"])
        else:
            self.model_dict["encoder"] = bert_utils.BERTMLMEncoder(
                checkpoint=init_params["checkpoint"],
                pool=True,
                use_bert_pooler=False if "use_bert_pooler" not in init_params else init_params["use_bert_pooler"])
        self.model_dict["model"] = bert_utils.BERTModel(self.model_dict["encoder"], num_tasks=len(init_params["task_targets"]))
        self.model_dict["model"].load_state_dict(torch.load(f"{model}/model.pth", map_location=self._device))
        self.model_dict["model"] = self.model_dict["model"].to(self._device).eval()
        return self.model_dict

    def predict(self, search_results):
        if not isinstance(search_results, list):
            raise ValueError("search_results should be a list of strings")
        print("[Encoding Data]")
        batches = chunks(search_results, self._batch_size)
        encodings = []
        for batch in batches:
            enc = self.tokenizer.batch_encode_plus(
                batch,
                add_special_tokens=True,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt")
            encodings.append(enc)
        predictions = []
        print("[Making Predictions]")
        for enc in encodings:
            inputs = {"input_ids": enc["input_ids"].to(self._device), "attention_mask": enc["attention_mask"].to(self._device)}
            outputs = self.model_dict["model"](inputs)
            logits = outputs[self.model_dict["task_id"]]
            logits = logits.detach().cpu().numpy()
            predictions.extend(logits)
        print("[Post Processing Results]")
        results = []
        targets = self.model_dict["targets"]
        for pred in predictions:
            res = {targets[i]: float(p) for i, p in enumerate(pred)}
            results.append(res)
        return results

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    stigma_search = StigmaSearch()
    stigma_results = stigma_search.search(text)
    
    # Formatting results
    formatted_results = []
    for result in stigma_results:
        formatted_results.append({
            "document_id": result["document_id"],
            "keyword_category": result["keyword_category"],
            "text": result["text"],
            "start": result["start"],
            "end": result["end"]
        })
    
    return jsonify({'results': formatted_results})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
