from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from demo.model.bart_xsum import produce_summary


def render_home_page(request):
    context = {}
    api_produce_summary("The W.H.O. said it needs $27.9 billion to speed production of a vaccine. Infections among Latinos in the U.S. have far outpaced those among the rest of the population during the recent surge in cases.")
    return render(request, 'index.html', context)


def api_produce_summary(source_txt):
    hypo = produce_summary(source_text=source_txt)
    print(hypo)