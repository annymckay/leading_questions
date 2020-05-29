import syntax
import argparse
import pickle


def get_syntax_bigrams(data):
    res = syntax.process_data(data)
    res = [r.syntax_bigrams for r in res]
    return res


def predict(questions):
    preprocessed = syntax.process_data(questions)
    preprocessed = [p.syntax_bigrams for p in preprocessed]

    vectors = tokenizer.texts_to_matrix(preprocessed, mode="binary")

    predictions = list(model.predict(vectors))

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-i', help='Режим получения вопроса через стандартный ввод', action='store_true')
    arg('-f', help='Режим получения вопросов из файла. F - путь до файла с вопросами.')

    args = parser.parse_args()

    print("Загрузка модели...")
    model = pickle.load(open(f"models/SVM_syntax_2000_classifier.pkl", "rb"))
    tokenizer = pickle.load(open(f"models/tokenizer.pkl", "rb"))

    answers = {
        0: "Нет",
        1: "Да"
    }

    if args.i:
        while True:
            print("Введите вопрос")
            question = input()
            prediction = predict([question])[0]
            print(answers[prediction])
    else:
        print("Получение вопросов")
        questions_file = args.f
        with open(questions_file, "r") as fp:
            questions = fp.readlines()
            questions = [q.strip("\n") for q in questions]

        print("Классификация...")
        predictions = predict(questions)

        with open("result.csv", "w") as fp:
            lines = [f"{questions[i]};{answers[p]}\n" for i, p in enumerate(predictions)]
            fp.writelines(lines)

        print("Результат сохранен в файл result.csv")







