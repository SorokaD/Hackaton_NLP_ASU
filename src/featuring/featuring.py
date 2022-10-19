from nltk.corpus import stopwords


def trans_lator(text):
    from googletrans import Translator
    translator = Translator()
    result = translator.translate(text, src='ru', dest='en')
    return result.text


# рабочая функция лематизации и приведения текстов
# stop_words = stopwords.words('english') + stopwords.words('russian')
def lemmatize(doc):
    import re
    import nltk
    from pymorphy2 import MorphAnalyzer
    from nltk.corpus import stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    patterns = r"0-9[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    morph = MorphAnalyzer()
    how_many_tokens = 5000  # количсетво первых слов которые будут возвращены
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stop_words:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    if len(tokens) >= 2:  # условие длинный слова !!!
        return tokens[0:how_many_tokens]  # str(' '.join(tokens[0:how_many_tokens]))
    return None

def transform(train_issues, train_comments, employees):
    import pandas as pd
    # Добавили столбец с количеством коментов
    train = train_issues.drop(columns=['created', 'key'], axis=1)
    train['lematize_summary'] = train_issues['summary'].apply(lemmatize).fillna('NotDefined')
    train = pd.merge(train, train_comments.groupby(by='issue_id').count()['text'], left_on="id", right_on="issue_id", how='left').fillna(0)
    train.rename(columns={'text':'comments'}, inplace=True)
    employees_data = employees.drop(columns=['active', 'full_name', 'salary_calculation_type', 'english_level']).fillna('NotDefined')
    employees_data = employees_data.drop(columns=['is_nda_signed', 'is_labor_contract_signed', 'is_added_one_to_one'], inplace=True)
    # добавим столбец с суммой комментариев
    train = pd.merge(train, train_comments.groupby('issue_id')['text'].apply(lambda x: ' '.join(x)), left_on="id", right_on="issue_id", how='left').fillna('NotDefined')
    train['lematize_comments'] = train['text'].apply(lemmatize).fillna('NotDefined')
    train.drop(columns='text', inplace=True)
    # добавили в train данные исполнителя, заказчика оставили просто как id
    train = pd.merge(train, employees_data.iloc[:,0:], left_on="assignee_id", right_on="id", how='left')
    # удалим лишние столбцы В ТОМ ЧИСЛЕ SUMMARY пока нет NLP модели
    train.drop(columns=['id_y', 'id_x',  'summary', 'assignee_id'], inplace=True)
    return(train)

def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []
    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features