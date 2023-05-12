import pandas as pd


def read_tokens(path, train=True):
    """
    Reads the file from the given path (txt file).
    Returns list tokens and list of labels if it is training file.
    Returns list of tokens if it is test file.
    """
    with open(path, 'r', encoding="utf-8") as f:
        data = f.read()

    if train:
        data = [[tuple(word.split('\t')) for word in instance.strip().split('\n')] for idx, instance in
                enumerate(data.split("SAMPLE_START\tO")) if len(instance) > 1]
        tokens = [[tupl[0].strip() for tupl in sent] for sent in data]
        labels = [[tupl[1] for tupl in sent] for sent in data]
        return tokens, labels
    else:
        tokens = [[word for word in instance.strip().split('\n')] for idx, instance in
                  enumerate(data.split("SAMPLE_START")) if len(instance) > 1]
        return tokens, None


def get_token_sentences(tokens, labels):
    sentence_count = 0
    sentence_without_trigger_count = 0
    token_data = []
    sentence_tokens = []
    sentence_token_labels = []
    instance_index = -1
    for temp_tokens, temp_labels in zip(tokens, labels):
        instance_index += 1

        SEP_indices = [i for i, value in enumerate(temp_tokens) if value == '[SEP]']

        if len(SEP_indices) == 0:  # If no [SEP] labels found, the instance has one sentence.
            sentence_count += 1
            sentence_tokens.append(temp_tokens)
            sentence_token_labels.append(temp_labels)
        else:
            SEP_indices.insert(0, -1)  # Add the index of -1 to the beginning
            SEP_indices.insert(len(SEP_indices), len(temp_tokens))

            sentence_token_list = []  # list of SentenceToken objects
            for i in range(0, len(SEP_indices)):
                if i < (len(SEP_indices) - 1):
                    temp_sent_tokens = temp_tokens[SEP_indices[i] + 1:SEP_indices[i + 1]]

                    temp_sent_labels = []
                    if len(temp_labels) > 0:
                        temp_sent_labels = temp_labels[SEP_indices[i] + 1:SEP_indices[i + 1]]
                        if "B-trigger" not in temp_sent_labels:
                            sentence_without_trigger_count += 1

                    sentence_count += 1
                    sentence_tokens.append(temp_sent_tokens)
                    sentence_token_labels.append(temp_sent_labels)

            token_data.append([temp_tokens, temp_labels, sentence_token_list])
    return sentence_tokens, sentence_token_labels, sentence_count, sentence_without_trigger_count


def format_token_data(input_path, output_path, train=True):
    tokens, labels = read_tokens(input_path, train=train)
    sentence_tokens, sentence_token_labels, token_sentence_count, sentence_without_trigger_count = get_token_sentences(
        tokens, labels)

    binary_token_labels = []
    for token_labels in sentence_token_labels:
        binary_token_labels.append([1 if label == 'B-trigger' or label == 'I-trigger' else 0 for label in token_labels])

    sentences = []
    for tokens in sentence_tokens:
        sentences.append(' '.join(tokens))

    df = pd.DataFrame(list(zip(sentences, sentence_tokens, sentence_token_labels, binary_token_labels)),
                      columns=['text', 'tokens', 'labels', 'rationales'])
    df.to_csv(output_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    input_path = 'CASE2021/subtask4-token/without_duplicates/en-train.txt'
    output_path = 'CASE2021/subtask4-token/without_duplicates/en-train.csv'
    format_token_data(input_path, output_path, train=True)
