from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('story', views.story, name='story'),
    path('news', views.news, name='news'),
    path('document', views.document, name='document'),
    path('handle_story', views.handle_story, name='handle_story'),
    path('handle_news', views.handle_news, name='handle_news'),
    path('handle_document', views.handle_document, name='handle_document'),
    path('eda', views.eda, name='eda'),
    path('model', views.model, name='model'),
    path('download-model/<str:filename>/', views.download_model, name='download_model'),
]

if settings.DEBUG:  
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)