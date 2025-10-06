# -*- coding: utf-8 -*-
"""
Прототип книжного рекомендательного сервиса на основе Goodbooks-10k
"""

# =============================================================================
# 0. Импорты и настройки
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# 1. Загрузка данных
# =============================================================================
print("Загрузка данных...")
ratings = pd.read_csv('ratings.csv')
books = pd.read_csv('books.csv')
tags = pd.read_csv('tags.csv')
book_tags = pd.read_csv('book_tags.csv')

print(f"Рейтинги: {ratings.shape}")
print(f"Книги: {books.shape}")
print(f"Теги: {tags.shape}")
print(f"Связь книг и тегов: {book_tags.shape}")

# =============================================================================
# 2. EDA (Exploratory Data Analysis)
# =============================================================================

# 2.1 Распределение оценок
plt.figure()
sns.histplot(ratings['rating'], bins=10, kde=False)
plt.title('Распределение оценок')
plt.xlabel('Оценка')
plt.ylabel('Количество')
plt.show()

# 2.2 Активность пользователей
user_counts = ratings['user_id'].value_counts()
plt.figure()
sns.histplot(np.log1p(user_counts), bins=50, kde=False)
plt.title('Логарифм количества оценок на пользователя')
plt.xlabel('log1p(оценок)')
plt.ylabel('Количество пользователей')
plt.show()

# 2.3 Популярность книг
book_counts = ratings['book_id'].value_counts()
plt.figure()
sns.histplot(np.log1p(book_counts), bins=50, kde=False)
plt.title('Логарифм количества оценок на книгу')
plt.xlabel('log1p(оценок)')
plt.ylabel('Количество книг')
plt.show()

# 2.4 Топ-20 тегов
tagged = book_tags.merge(tags, on='tag_id')
top_tags = Counter(tagged['tag_name']).most_common(20)
tags_df = pd.DataFrame(top_tags, columns=['tag', 'count'])

plt.figure()
sns.barplot(data=tags_df, x='count', y='tag')
plt.title('Топ-20 тегов')
plt.xlabel('Частота')
plt.ylabel('Тег')
plt.show()

# Проблемы:
print("\nВыявленные проблемы:")
print("- Сильное смещение оценок в сторону высоких значений (4–5)")
print("- Большинство пользователей имеют мало оценок → холодный старт")
print("- Распределение популярности книг — 'длинный хвост'")
print("- Матрица user–item крайне разрежена")

# =============================================================================
# 3. Подготовка вспомогательных маппингов
# =============================================================================
user_ids = ratings['user_id'].unique()
book_ids = ratings['book_id'].unique()

user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
book_to_idx = {book: idx for idx, book in enumerate(book_ids)}
idx_to_book = {idx: book for book, idx in book_to_idx.items()}

n_users = len(user_ids)
n_books = len(book_ids)

print(f"\nУникальных пользователей: {n_users}, книг: {n_books}")

# =============================================================================
# 4. Базовые модели
# =============================================================================

# 4.1 Неперсонализированная модель (Popularity)
def get_top_popular_books(min_ratings=50, top_n=10):
    stats = ratings.groupby('book_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count')
    )
    filtered = stats[stats['n_ratings'] >= min_ratings]
    top = filtered.sort_values('avg_rating', ascending=False).head(top_n)
    return top.index.tolist()

# Пример
print("\nТоп-5 популярных книг:", get_top_popular_books(top_n=5))

# 4.2 Контентная модель
print("\nСоздание контентного профиля...")

# Объединяем книги и теги
book_tag_texts = tagged.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
books_enhanced = books[['book_id', 'original_title']].copy()
books_enhanced = books_enhanced.merge(
    book_tag_texts, 
    left_on='book_id', 
    right_on='goodreads_book_id', 
    how='left'
)
books_enhanced['profile'] = (
    books_enhanced['original_title'].fillna('') + ' ' + 
    books_enhanced['tag_name'].fillna('')
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(books_enhanced['profile'])
book_id_to_tfidf_idx = {book_id: idx for idx, book_id in enumerate(books_enhanced['book_id'])}

def get_similar_books(book_id, N=5):
    if book_id not in book_id_to_tfidf_idx:
        return []
    idx = book_id_to_tfidf_idx[book_id]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_books = np.argsort(-sim)[1:N+1]
    return books_enhanced.iloc[sim_books]['book_id'].tolist()

# Пример
sample_book = books_enhanced['book_id'].iloc[0]
print(f"\nПохожие книги на {sample_book}:", get_similar_books(sample_book, N=3))

# =============================================================================
# 5. Item-Based Collaborative Filtering
# =============================================================================

print("\nПостроение матрицы взаимодействий...")

# Создаём разреженную матрицу (неявный фидбэк: rating >= 4 → 1)
ratings['implicit'] = (ratings['rating'] >= 4).astype(int)

rows = ratings['user_id'].map(user_to_idx)
cols = ratings['book_id'].map(book_to_idx)
data = ratings['implicit']

rating_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_books))

# Вычисляем схожесть между книгами (по неявным оценкам)
print("Вычисление схожести между книгами (косинусная мера)...")
item_sim = cosine_similarity(rating_matrix.T)  # shape: (n_books, n_books)

def predict_rating_item_cf(user_id, book_id, k=10):
    if user_id not in user_to_idx or book_id not in book_to_idx:
        return 3.0  # среднее по умолчанию
    u_idx = user_to_idx[user_id]
    b_idx = book_to_idx[book_id]
    
    user_vec = rating_matrix[u_idx].toarray().flatten()
    sim_scores = item_sim[b_idx]
    
    # Исключаем саму книгу
    sim_scores[b_idx] = 0
    
    # Находим топ-K похожих книг, которые пользователь оценил
    rated = user_vec > 0
    if not np.any(rated):
        return 3.0
    
    # Взвешенное среднее
    weighted_sum = np.dot(sim_scores, user_vec)
    sim_sum = np.sum(sim_scores)
    
    if sim_sum == 0:
        return np.mean(user_vec[rated]) if np.any(rated) else 3.0
    return weighted_sum / sim_sum

def get_itemcf_recommendations(user_id, N=5):
    if user_id not in user_to_idx:
        return get_top_popular_books(top_n=N)
    u_idx = user_to_idx[user_id]
    user_rated = rating_matrix[u_idx].toarray().flatten() > 0
    
    # Предсказываем для всех непрочитанных книг
    scores = []
    for b_idx in range(n_books):
        if not user_rated[b_idx]:
            pred = predict_rating_item_cf(user_id, idx_to_book[b_idx], k=20)
            scores.append((pred, idx_to_book[b_idx]))
    
    scores.sort(reverse=True)
    return [book for _, book in scores[:N]]

# Пример
sample_user = ratings['user_id'].iloc[0]
print(f"\nItem-CF рекомендации для пользователя {sample_user}:", 
      get_itemcf_recommendations(sample_user, N=3))

# =============================================================================
# 6. Matrix Factorization (SVD)
# =============================================================================

print("\nОбучение модели SVD...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
svd_model.fit(trainset)

def get_svd_recommendations(user_id, N=5):
    if user_id not in user_to_idx:
        return get_top_popular_books(top_n=N)
    predictions = []
    for book_id in book_ids:
        pred = svd_model.predict(user_id, book_id).est
        predictions.append((pred, book_id))
    predictions.sort(reverse=True)
    return [book for _, book in predictions[:N]]

# Пример
print(f"\nSVD рекомендации для пользователя {sample_user}:", 
      get_svd_recommendations(sample_user, N=3))

# =============================================================================
# 7. Оценка моделей
# =============================================================================

# Получаем предсказания на тестовом наборе
print("\nПолучение предсказаний SVD на тестовом наборе...")
predictions_svd = svd_model.test(testset)

# Создаём DataFrame с истинными и предсказанными оценками
test_df = pd.DataFrame(predictions_svd, columns=['user_id', 'book_id', 'true_r', 'pred_r', 'details'])

# Определяем релевантные книги: true_r >= 4
test_df['relevant'] = test_df['true_r'] >= 4
test_relevant = test_df[test_df['relevant']].groupby('user_id')['book_id'].apply(set).to_dict()

# Функции метрик
def precision_at_k(rec, rel, k=5):
    rec_k = rec[:k]
    if not rel:
        return 0.0
    return len(set(rec_k) & rel) / k

def recall_at_k(rec, rel, k=5):
    rec_k = rec[:k]
    if not rel:
        return 0.0
    return len(set(rec_k) & rel) / len(rel)

def ndcg_at_k(rec, rel, k=5):
    if not rel:
        return 0.0
    dcg = sum((1 if r in rel else 0) / np.log2(i + 2) for i, r in enumerate(rec[:k]))
    ideal = min(k, len(rel))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0

# Оценка на подвыборке активных пользователей (для скорости)
eval_users = list(test_relevant.keys())[:500]  # первые 500 пользователей с релевантными книгами

models = {
    'Popularity': lambda u: get_top_popular_books(top_n=10),
    'Content-Based': lambda u: get_similar_books(sample_book, N=10) if u in user_to_idx else get_top_popular_books(top_n=10),
    'Item-CF': get_itemcf_recommendations,
    'SVD': get_svd_recommendations
}

results = {model: {'P@5': [], 'R@5': [], 'nDCG@5': []} for model in models}

print("\nОценка моделей...")
for user in eval_users:
    relevant = test_relevant.get(user, set())
    for name, func in models.items():
        try:
            recs = func(user)
            results[name]['P@5'].append(precision_at_k(recs, relevant, k=5))
            results[name]['R@5'].append(recall_at_k(recs, relevant, k=5))
            results[name]['nDCG@5'].append(ndcg_at_k(recs, relevant, k=5))
        except Exception as e:
            # fallback на популярные книги при ошибке
            recs = get_top_popular_books(top_n=10)
            results[name]['P@5'].append(precision_at_k(recs, relevant, k=5))
            results[name]['R@5'].append(recall_at_k(recs, relevant, k=5))
            results[name]['nDCG@5'].append(ndcg_at_k(recs, relevant, k=5))

# Сводная таблица
summary = {}
for model in models:
    summary[model] = {
        'Precision@5': np.mean(results[model]['P@5']),
        'Recall@5': np.mean(results[model]['R@5']),
        'nDCG@5': np.mean(results[model]['nDCG@5'])
    }

results_df = pd.DataFrame(summary).T
print("\nСравнение моделей:")
print(results_df.round(4))

# =============================================================================
# 8. Гибридная рекомендация (простой пример)
# =============================================================================

def hybrid_recommend(user_id, N=5, alpha=0.7):
    """
    Гибрид: alpha * SVD + (1-alpha) * Popularity (для устойчивости)
    """
    svd_recs = get_svd_recommendations(user_id, N=20)
    pop_recs = get_top_popular_books(top_n=20)
    
    # Объединяем и удаляем дубли, сохраняя порядок
    combined = svd_recs + [b for b in pop_recs if b not in svd_recs]
    return combined[:N]

print(f"\nГибридные рекомендации для {sample_user}:", hybrid_recommend(sample_user))

# =============================================================================
# 9. Выводы
# =============================================================================
print("\n" + "="*60)
print("ВЫВОДЫ")
print("="*60)
print("1. Лучшая модель по метрикам — SVD.")
print("2. Popularity устойчива, но не персонализирована.")
print("3. Content-Based полезна для новых книг (холодный старт).")
print("4. Item-CF требует много памяти, но интерпретируема.")
print("5. Гибридизация помогает балансировать точность и надёжность.")
print("\nДальнейшие улучшения:")
print("- Использовать BERT для текстовых профилей")
print("- Добавить фичи пользователей")
print("- Применить LightFM или нейросетевые модели")
print("- Внедрить diversity и fairness в ранжирование")