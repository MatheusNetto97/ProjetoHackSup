# bibliotecas usadas
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV


class SupportTicket:
    def __init__(self, problem, urgency):
        self.problem = problem
        self.urgency = urgency


class SupportTeam:
    def __init__(self, name, expertise):
        self.name = name
        self.expertise = expertise


class SupportSystem:
    def __init__(self, teams):
        self.teams = teams
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.text_vectorizer = CountVectorizer(
            tokenizer=word_tokenize, stop_words=nltk.corpus.stopwords.words('portuguese'))
        self.text_classifier = MultinomialNB()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def train_classifier(self, training_data):
        # Preparar os dados de treinamento para o classificador de problemas
        problem_texts = [ticket.problem for ticket in training_data]
        labels = [ticket.urgency for ticket in training_data]
        problem_vectors = self.vectorizer.fit_transform(problem_texts)
        self.classifier.fit(problem_vectors, labels)

        # Preparar os dados de treinamento para o classificador de texto
        texts = [data['ticket'] for data in training_data_text]
        categories = [data['category'] for data in training_data_text]
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        text_vectors = self.text_vectorizer.fit_transform(preprocessed_texts)

        # Ajuste de hiperparâmetros para o classificador de texto
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'fit_prior': [True, False]
        }
        self.text_classifier = GridSearchCV(
            self.text_classifier, param_grid, cv=3)
        self.text_classifier.fit(text_vectors, categories)

    def predict_category(self, problem_text):
        problem_vector = self.vectorizer.transform([problem_text])
        predicted_category = self.classifier.predict(problem_vector)
        return predicted_category[0]

    def predict_text_category(self, text):
        preprocessed_text = self.preprocess_text(text)
        text_vector = self.text_vectorizer.transform([preprocessed_text])
        predicted_category = self.text_classifier.predict(text_vector)
        return predicted_category[0]

    def preprocess_text(self, text):
        # Remoção de caracteres especiais e pontuações
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)

        # Lematização ou stemming
        # text = self.lemmatizer.lemmatize(text)
        text = self.stemmer.stem(text)

        return text

    def escalate_ticket(self, ticket):
        problem_category = self.predict_category(ticket.problem)
        relevant_teams = []
        for team in self.teams:
            if problem_category in team.expertise:
                relevant_teams.append(team)

        if relevant_teams:
            selected_team = random.choice(relevant_teams)
            print(
                f"Ticket '{ticket.problem}' escalated to support team '{selected_team.name}'")
        else:
            print(f"No support team available for ticket '{ticket.problem}'")


# Equipes de suporte
team1 = SupportTeam("Team 1", ["software development", "bug fixing"])
team2 = SupportTeam("Team 2", ["hardware troubleshooting", "device setup"])
team3 = SupportTeam(
    "Team 3", ["network configuration", "internet connectivity"])

# Sistema de suporte
support_system = SupportSystem([team1, team2, team3])

# Dados de treinamento para o classificador de problemas
training_data = [
    SupportTicket("I'm having trouble installing the software", "high"),
    SupportTicket("The hardware device is not working", "medium"),
    SupportTicket("I can't connect to the internet", "high"),
    # Adicione mais exemplos de dados de treinamento aqui
]

# Dados de treinamento para o classificador de texto
training_data_text = [
    {'ticket': 'Problema na conexão com a internet', 'category': 'Alta'},
    {'ticket': 'Problema ao fazer login no sistema', 'category': 'Média'},
    {'ticket': 'Dúvida sobre a funcionalidade X', 'category': 'Baixa'},
    # Adicione mais exemplos de tickets e suas respectivas categorias
]

# Treinar os classificadores
support_system.train_classifier(training_data)

# Exemplo de chamado de suporte
ticket = SupportTicket("I'm experiencing a network issue", "medium")

# Escalonar o chamado para a equipe adequada
support_system.escalate_ticket(ticket)

# Exemplo de classificação de um novo ticket de texto
new_ticket = 'Problema com o envio de e-mails'
predicted_category = support_system.predict_text_category(new_ticket)
print('Categoria prevista para o ticket de texto:', predicted_category)
