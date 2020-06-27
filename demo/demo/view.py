from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from demo.model import bart_xsum, bart_cnn
from django.views.decorators.csrf import csrf_exempt

# Update this to show which models are available in the demo
AVAILABLE_MODELS = [bart_xsum, bart_cnn]

AVAILABLE_MODELS_MAPPING = {m.MODEL_NAME: m for m in AVAILABLE_MODELS}
AVAILABLE_MODELS_NAME = [m.MODEL_NAME for m in AVAILABLE_MODELS]


def render_home_page(request):
    context = {
        "models": AVAILABLE_MODELS_NAME
    }

    return render(request, 'index.html', context)


@csrf_exempt
def api_produce_summary(request):
    if request.method != 'POST':
        return HttpResponse(status=405)

    source = request.POST.get('source')
    model_type = request.POST.get('model_type')

    _model = AVAILABLE_MODELS_MAPPING[model_type]

    hypo = _model.produce_summary(source_text=source)

    return JsonResponse({
        'summary': hypo
    })