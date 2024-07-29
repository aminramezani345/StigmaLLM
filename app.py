
import os
import json
import torch
from transformers import BertTokenizer
from flask import Flask, render_template_string, request, redirect, url_for
import threading
import webbrowser
from PyPDF2 import PdfReader

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

    def format_for_model(self, keyword_category, search_results):
        subset = list(filter(lambda i: i["keyword_category"] == keyword_category, search_results))
        if len(subset) == 0:
            print(">> WARNING: No matches for the specified keyword category were found in the search results.")
            return (None, None, None)
        return ([s["document_id"] for s in subset], [s["keyword"] for s in subset], [s["text"] for s in subset])


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

template = """
<!DOCTYPE html>
<html>
<head>
    <title>Stigma Highlighting</title>
    <style>
        body {
            background-color: #f0f0f0; /* Change background color */
        }
        .highlight-keyword {
            background-color: red; /* Highlight keyword in red */
            color: white; /* Text color for readability */
            font-weight: bold; /* Bold text for emphasis */
        }
        .highlight-text {
            background-color: yellow; /* Highlight surrounding text in yellow */
        }
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            padding: 8px;
        }
        th { background-color: #f2f2f2; }
    </style>
    <script>
        function highlightText(event, start, end) {
            event.preventDefault();

            var textarea = document.getElementById('main_text');
            var text = textarea.value;

            // Clear previous highlighting
            var newText = text.replace(/<span[^>]*>/g, '');

            // Apply new highlighting
            newText = newText.substring(0, start-20) +
                      '<span class="highlight-text">' + 
                      text.substring(start-20, start-1) + 
                      '</span>'+ 
                      '<span class="highlight-keyword">' + 
                      text.substring(start-1, start+7) + 
                      '</span>' + 
                      '<span class="highlight-text">' + 
                      text.substring(start+7, end+30) + 
                      '</span>'+
                      newText.substring(end+20);

            // Update the displayed text container with HTML content
            document.getElementById('highlighted_text_container').innerHTML = newText;
        }
    </script>
</head>
<body>
    <h1 style="text-align: center;">Stigma Highlighting Tool</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="file">Upload a text or PDF file:</label><br>
        <input type="file" id="file" name="file"><br><br>
        <button type="submit" name="upload_file">Upload File</button>
    </form>
    <textarea id="main_text" style="display:none;">{{ main_text|safe }}</textarea>
    <div id="highlighted_text_container" style="white-space: pre-wrap;">{{ main_text|safe }}</div>
    {% if stigma_results %}
        <h2>Search Results:</h2>
        <table>
            <tr>
                <th>Keyword</th>
                <th>Keyword_Category</th>
                <th>Context</th>
                <th>Action</th>
            </tr>
            {% for result in stigma_results %}
                <tr>
                    <td>{{ result.keyword }}</td>
                    <td>{{ result.keyword_category }}</td>
                    <td>{{ result.text }}</td>
                    <td>
                        <button onclick="highlightText(event, {{ result.start }}, {{ result.end }})">Highlight</button>
                    </td>
                </tr>
            {% endfor %}
        </table>
    {% endif %}
    {% if predictions %}
        <h2>Predictions:</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Score</th>
            </tr>
            {% for prediction in predictions %}
                <tr>
                    <td>{{ prediction[0] }}</td>
                    <td>{{ prediction[1] }}</td>
                </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            if file.filename.endswith('.txt'):
                main_text = file.read().decode('utf-8')
            elif file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                main_text = ''.join(page.extract_text() for page in pdf_reader.pages)
            else:
                return "Unsupported file format"

            stigma_search = StigmaSearch()
            stigma_results = stigma_search.search(main_text)
            return render_template_string(template, main_text=main_text, stigma_results=stigma_results)
    return render_template_string(template, main_text="")

if __name__ == "__main__":
    port = 5000 + os.getpid()
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
