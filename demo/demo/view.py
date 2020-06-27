from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from demo.model import bart_xsum, bart_cnn
from django.views.decorators.csrf import csrf_exempt


def render_home_page(request):
    
    context = {
        "models"
    }

    return render(request, 'index.html', context)


@csrf_exempt
def api_produce_summary(request):
    if request.method != 'POST':
        return HttpResponse(status=405)

    source = request.POST.get('source')
    hypo = bart_cnn.produce_summary(source_text=source)

    return JsonResponse({
        'summary': hypo
    })