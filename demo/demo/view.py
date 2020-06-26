from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from demo.model.bart_xsum import produce_summary


def render_home_page(request):
    context = {}
    return render(request, 'index.html', context)


def api_produce_summary(request):
    if request.method != 'POST':
        return HttpResponse(status=405)

    source = request.post.get('source')
    hypo = produce_summary(source_text=source)

    return JsonResponse({
        'summary': hypo
    })