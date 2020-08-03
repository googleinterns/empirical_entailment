from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import importlib

# name of model classes from demo/model/
AVAILABLE_MODELS = ["bart_base", "bart_base_entailment", "bart_base_vocab_constraint"]
AVAILABLE_MODEL_CLASSES = []

for _model in AVAILABLE_MODELS:
    _model = '.' + _model
    AVAILABLE_MODEL_CLASSES.append(importlib.import_module(_model, 'demo.model'))

AVAILABLE_MODELS_MAPPING = {m.MODEL_NAME: m for m in AVAILABLE_MODEL_CLASSES}
AVAILABLE_MODELS_NAME = [m.MODEL_NAME for m in AVAILABLE_MODEL_CLASSES]


def render_home_page(request):
    """
    Renders the homepage for the demo.
    :param request:
    :return:
    """
    context = {
        "models": AVAILABLE_MODELS_NAME
    }

    return render(request, 'index.html', context)


@csrf_exempt
def api_produce_summary(request) -> JsonResponse:
    """
    API for using available models to produce summary.

    :param request: a POST request containing the following parameters
        "source": source text
        "model_type": name of the model to used; model names are specified in each model files under model/

    :return: A json response containing the following items
        "summary": produced summary
    """
    if request.method != 'POST':
        return HttpResponse(status=405)

    source = request.POST.get('source')
    model_type = request.POST.get('model_type')

    _model = AVAILABLE_MODELS_MAPPING[model_type]

    hypo = _model.produce_summary(source_text=source)

    return JsonResponse({
        'summary': hypo
    })