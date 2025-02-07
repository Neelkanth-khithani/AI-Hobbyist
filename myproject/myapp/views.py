from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from package.idea.text_processing import clean, extract_pos
from package.idea.problem_statement import ai_generate_problem_statement, nlg_generate_problem_statement
from package.idea.fetch_news import fetch_sustainability_news
from package.idea.rag import *
from package.idea.eda import *
from io import BytesIO
from django.core.files.base import ContentFile
from matplotlib import pyplot as plt
import seaborn as sns
from django.conf import settings 
import matplotlib
matplotlib.use('Agg')
from package.idea.model import *

def index(request):
    return render(request, 'index.html')

def story(request):
    return render(request, 'story.html', {"story": None})

def news(request):
    return render(request, 'news.html')

def document(request):
    return render(request, 'document.html')

def eda(request):
    return render(request, 'eda.html')

def handle_story(request):
    context = {
        'word_count': None,
        'story_text': None,
        'error': None,
        'problem_statement': None
    }
    
    if request.method == "POST":
        if 'story' in request.POST:
            story_text = request.POST.get("story", "").strip()
            if not story_text:
                context['error'] = "Story text cannot be empty."
                return render(request, 'handle_story.html', context)
            
            cleaned_story = clean(story_text)
            words = extract_pos(cleaned_story, pos_types=("NN",))
            context.update({
                'word_count': {word: cleaned_story.split().count(word) for word in set(words)},
                'story_text': story_text
            })
        
        elif all(key in request.POST for key in ['selected_words', 'task', 'ps_choice']):
            selected_words = request.POST.getlist("selected_words")
            task_choice = request.POST.get("task")
            ps_choice = request.POST.get("ps_choice")
            
            if not selected_words:
                context['error'] = "Please select at least one word."
                return render(request, 'handle_story.html', context)
            
            try:
                if ps_choice == "nlg":
                    problem_statement = nlg_generate_problem_statement(selected_words, task_choice)
                elif ps_choice == "ai":
                    problem_statement = ai_generate_problem_statement(selected_words, task_choice)
                else:
                    context['error'] = "Invalid problem statement generation choice."
                    return render(request, 'handle_story.html', context)
                    
                context['problem_statement'] = problem_statement
            except Exception as e:
                context['error'] = f"Error generating problem statement: {str(e)}"
    
    return render(request, 'handle_story.html', context)

def handle_news(request):
    context = {
        'articles': None,
        'word_count': None,
        'selected_article': None,
        'error': None,
        'problem_statement': None
    }
    
    if request.method == "GET":
        articles = fetch_sustainability_news()
        if not articles:
            context['error'] = "Failed to fetch news articles. Please try again."

        request.session['articles'] = articles
        context['articles'] = articles
        
    elif request.method == "POST":
        articles = request.session.get('articles', [])
        context['articles'] = articles
        
        if 'article_index' in request.POST:
            article_index = int(request.POST.get('article_index'))
            if 0 <= article_index < len(articles):
                title, description = articles[article_index]
                full_text = f"{title}. {description}"
                
                cleaned_text = clean(full_text)
                words = extract_pos(cleaned_text, pos_types=("NN",))
                context.update({
                    'word_count': {word: cleaned_text.split().count(word) for word in set(words)},
                    'selected_article': full_text
                })
            else:
                context['error'] = "Invalid article selection."
                
        elif all(key in request.POST for key in ['selected_words', 'task', 'ps_choice']):
            selected_words = request.POST.getlist("selected_words")
            task_choice = request.POST.get("task")
            ps_choice = request.POST.get("ps_choice")
            
            if not selected_words:
                context['error'] = "Please select at least one word."
                return render(request, 'handle_news.html', context)
            
            try:
                if ps_choice == "nlg":
                    problem_statement = nlg_generate_problem_statement(selected_words, task_choice)
                elif ps_choice == "ai":
                    problem_statement = ai_generate_problem_statement(selected_words, task_choice)
                else:
                    context['error'] = "Invalid problem statement generation choice."
                    return render(request, 'handle_news.html', context)
                    
                context['problem_statement'] = problem_statement
            except Exception as e:
                context['error'] = f"Error generating problem statement: {str(e)}"
    
    return render(request, 'handle_news.html', context)

def handle_document(request):
    if request.method == "GET":
        return redirect('document')
        
    context = {
        'word_count': None,
        'document_text': None,
        'summary': None,
        'error': None,
        'problem_statement': None
    }
    
    if request.method == "POST":
        if 'document' in request.FILES:
            try:
                uploaded_file = request.FILES['document']
                if not uploaded_file.name.endswith('.pdf'):
                    return render(request, 'document.html', {'error': "Please upload a PDF file."})
                
                fs = FileSystemStorage()
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                
                document_text = extract_text_from_pdf(file_path)
                documents = [document_text]
                document_embeddings = generate_embeddings(documents)
                retrieved_document = retrieve_top_k_documents(document_embeddings, documents)[0]
                summary = generate_answer(retrieved_document)
                
                fs.delete(filename)
                
                cleaned_document = clean(summary)
                words = extract_pos(cleaned_document, pos_types=("NN",))
                context.update({
                    'word_count': {word: cleaned_document.split().count(word) for word in set(words)},
                    'document_text': document_text,
                    'summary': summary
                })
                
                return render(request, 'handle_document.html', context)
                
            except Exception as e:
                return render(request, 'document.html', {'error': f"Error processing document: {str(e)}"})
                
        elif all(key in request.POST for key in ['selected_words', 'task', 'ps_choice']):
            selected_words = request.POST.getlist("selected_words")
            task_choice = request.POST.get("task")
            ps_choice = request.POST.get("ps_choice")
            
            if not selected_words:
                context['error'] = "Please select at least one word."
                return render(request, 'handle_document.html', context)
            
            try:
                if ps_choice == "nlg":
                    problem_statement = nlg_generate_problem_statement(selected_words, task_choice)
                elif ps_choice == "ai":
                    problem_statement = ai_generate_problem_statement(selected_words, task_choice)
                else:
                    context['error'] = "Invalid problem statement generation choice."
                    return render(request, 'handle_document.html', context)
                    
                context['problem_statement'] = problem_statement
                return render(request, 'handle_document.html', context)
            except Exception as e:
                context['error'] = f"Error generating problem statement: {str(e)}"
                return render(request, 'handle_document.html', context)
    
    return render(request, 'document.html')

def eda(request):
    dataset = None
    error = None
    data_types = None
    missing_data_summary = {}
    categorical_summary = {}
    outlier_visualizations = {}
    scaling_details = {}
    no_scaling_required = False
    show_categorical = False
    no_outliers_detected = True 
    context = {}

    if request.method == 'POST':
        dataset_text = request.POST.get('dataset', '').strip()
        target_column = request.POST.get('target_column', '').strip()

        if dataset_text:
            try:
                dataset = convert_text_to_dataframe(dataset_text)

                if dataset is not None and not dataset.empty:
                    dataset = remove_contiguous_columns(dataset)
                    dataset, notes = handle_missing_values(dataset)
                    missing_data_summary = get_missing_data_summary(dataset, notes)
                    dataset, categorical_summary = handle_categorical_encoding(dataset, target_column)
                    data_types = [(col, str(dataset[col].dtype)) for col in dataset.columns]

                    dataset, visualization_details = handle_outliers_auto(dataset)
                    no_outliers_detected = len(visualization_details) == 0  

                    for col, (data, outliers, title) in visualization_details.items():
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=data[col], whis=1.5)
                        plt.title(f"{title}: Outliers in {col}")
                        plt.xlabel(col)
                        plt.tight_layout()

                        image_name = f"{col}_outliers.png"
                        image_path = os.path.join(settings.MEDIA_ROOT, image_name)
                        plt.savefig(image_path, format="png")
                        plt.close()

                        outlier_visualizations[col] = {
                            "method": title,
                            "outliers_removed": len(outliers),
                            "image_path": os.path.join(settings.MEDIA_URL, image_name),  # Use MEDIA_URL for frontend access
                        }

                    dataset, scaling_details = handle_scaling_auto(dataset)
                    no_scaling_required = all(
                        details['scaling_method'] == 'None' for details in scaling_details.values()
                    )

                    show_categorical = True
                else:
                    error = "Dataset is empty or could not be processed."
            except Exception as e:
                error = f"Error processing dataset: {str(e)}"
        else:
            error = "No dataset provided."

    context.update({
        'dataset': dataset if dataset is not None and not dataset.empty else None,
        'error': error,
        'data_types': data_types,
        'missing_data_summary': missing_data_summary,
        'categorical_summary': categorical_summary,
        'outlier_visualizations': outlier_visualizations,
        'scaling_details': scaling_details,
        'no_scaling_required': no_scaling_required,
        'show_categorical': show_categorical,
        'no_outliers_detected': no_outliers_detected  # Pass this to the context
    })

    return render(request, 'eda.html', context)


def get_missing_data_summary(dataset, notes=None):
    summary = {}
    for col in dataset.columns:
        summary[col] = {
            'null_values': dataset[col].isnull().sum(),
            'missing_percentage': (dataset[col].isnull().mean() * 100),
            'note': notes.get(col, 'No action required') if notes else ''
        }
    return summary

def model(request):
    if request.method == "POST":
        try:
            dataset = request.FILES['dataset']
            dataset_str = dataset.read().decode('utf-8')
            target_variable = request.POST['target_variable']
            model_type = request.POST['model_type']
            problem_type = request.POST['problem_type']
            
            # Get and validate model name
            model_name = request.POST.get('model_name', '').strip()
            if not model_name:
                raise ValueError("Model name is required")
            
            # Remove special characters and spaces from model name
            model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_'))
            
            X, y, dataset_df, feature_names = process_data(dataset_str, target_variable)
            
            stratify = y if problem_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            
            preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            if problem_type == "regression":
                model, metrics = train_and_evaluate_regression(
                    preprocessor, X_train, X_test, y_train, y_test, model_type
                )
                
                filename = save_model(model, model_name)
                
                context = {
                    'rms_error': metrics['rmse'],
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'mape': metrics['mape'],
                    'cv_score': metrics['cv_score'],
                    'dataset': dataset_df.to_html(classes="table table-striped"),
                    'model_type': model_type,
                    'problem_type': problem_type,
                    'feature_importances': get_feature_importances(model, feature_names),
                    'model_filename': filename
                }
            
            elif problem_type == "classification":
                model, metrics = train_and_evaluate_classification(
                    preprocessor, X_train, X_test, y_train, y_test, model_type
                )
                
                filename = save_model(model, model_name)
                
                context = {
                    'cm': metrics['confusion_matrix'],
                    'f1': metrics['f1_score'],
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'class_report': metrics['classification_report'],
                    'dataset': dataset_df.to_html(classes="table table-striped"),
                    'model_type': model_type,
                    'problem_type': problem_type,
                    'model_filename': filename
                }
            
            return render(request, 'model.html', context)
        
        except Exception as e:
            return render(request, 'model.html', {'error': str(e)})
    
    return render(request, 'model.html')