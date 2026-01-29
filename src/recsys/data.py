# -*- coding: utf-8 -*-
"""
Загрузка и подготовка данных Goodbooks-10k.
Читает CSV: ratings, books, tags, book_tags.
"""
from typing import Tuple

import pandas as pd

from .config import (
    PATH_RATINGS,
    PATH_BOOKS,
    PATH_TAGS,
    PATH_BOOK_TAGS,
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает все четыре CSV-файла датасета.

    Returns
    -------
    ratings : pd.DataFrame
        Колонки: user_id, book_id, rating.
    books : pd.DataFrame
        Метаданные книг (id, book_id, original_title, ...).
    tags : pd.DataFrame
        tag_id, tag_name.
    book_tags : pd.DataFrame
        goodreads_book_id, tag_id, count.
    """
    ratings = pd.read_csv(PATH_RATINGS)
    books = pd.read_csv(PATH_BOOKS)
    tags = pd.read_csv(PATH_TAGS)
    book_tags = pd.read_csv(PATH_BOOK_TAGS)
    return ratings, books, tags, book_tags


def build_book_tag_merged(
    books: pd.DataFrame,
    tags: pd.DataFrame,
    book_tags: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединяет книги с тегами для контентной модели.
    goodreads_book_id в book_tags соответствует book_id в books.

    Returns
    -------
    pd.DataFrame
        book_id, original_title, tag_text, profile (title + tags).
    """
    tagged = book_tags.merge(tags, on='tag_id')
    tag_agg = (
        tagged.groupby('goodreads_book_id')['tag_name']
        .apply(lambda x: ' '.join(x))
        .reset_index()
    )
    tag_agg.columns = ['goodreads_book_id', 'tag_text']

    books_sub = books[['book_id', 'original_title']].copy()
    merged = books_sub.merge(
        tag_agg,
        left_on='book_id',
        right_on='goodreads_book_id',
        how='left',
    )
    merged['profile'] = (
        merged['original_title'].fillna('') + ' ' + merged['tag_text'].fillna('')
    )
    return merged

