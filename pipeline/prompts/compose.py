from pipeline.prompts.examples import examples
from pipeline.prompts.facts import inline
from pipeline.prompts.guidelines import common, natural


# common messages used in the prompt -- independent of KG facts
common_messages = {
    'system': common.system_message.lstrip(),
    'retry': common.retry_message.lstrip(),
    'retry_mechanism': common.retry_mechanism.lstrip()
}


# guideline messages used in the prompt
guidelines = {
    'correctness': natural.correctness_guidelines.lstrip(),
    'other': natural.other.lstrip(),
    'reminder': natural.retry_reminder.lstrip()
}


# fact-specific messages used in the prompt
facts = {
    'description': inline.description.lstrip(),
    'format': inline.fact.lstrip()
}


# fewshot examples used in the prompt
fewshot_examples = {
    'NELL': examples.sport_examples,
    'OTHER': examples.general_examples
}


def format_examples(example_list, dataset):
    """
    Format fewshot examples to use within prompts.

    :param example_list: list of examples to format
    :param dataset: the dataset considered to perform annotations with LLMs
    :return: formatted examples
    """

    if 'YAGO4.5' in dataset:  # when working w/ YAGO, predicate examples are stored w/ 'YAGO' as key -- reset dataset name
        dataset = 'YAGO'

    # format the examples as in the prompt config
    formatted_exs = '\n### Examples\n'
    for ex in example_list:
        formatted_exs += facts['format'][2:].rstrip().format(
            s=ex['fact']['s'],
            p=ex['fact']['p'] if dataset == 'NELL' else ex['fact']['p'][dataset],  # NELL doesn't have the inner dict for predicate
            o=ex['fact']['o'],
        ) + ' ' + ex['response']['correctness'] + '\n\n'

    return formatted_exs.replace('{', '{{').replace('}', '}}')


def compose_prompts(is_dbpedia, dataset, max_retries=5):
    """
    Compose the prompts associated w/ the KG facts

    :param is_dbpedia: True if the chosen dataset is DBPEDIA, False otherwise
    :param dataset: the target dataset
    :param max_retries: maximum number of LLM retries
    :return: the prompts composed for the KG facts
    """

    # define placeholder values for DBPEDIA dataset
    dbp = 'Your knowledge is limited to the year 2015. ' if is_dbpedia else ''
    dbp_info = ' up to 2015' if is_dbpedia else ''

    # define placeholder values for the prompts
    labels = '"T", "F"'
    responses = '"T" (true), or "F" (false)'

    # example formatting
    example_list = fewshot_examples['NELL'] if dataset == 'NELL' else fewshot_examples['OTHER']
    formatted_examples = format_examples(example_list, dataset)

    # system prompt composition
    system = (
        common_messages['system'] + dbp + '\n\n' +
        guidelines['correctness'] + '\n' +
        guidelines['other'] + '\n' +
        common_messages['retry_mechanism'] +
        formatted_examples
    )

    # user prompt composition
    user = (
        facts['description'] + '\n' +
        facts['format']
    )

    # retry prompt composition
    retry = (
        common_messages['retry'] + '\n' +
        guidelines['reminder'] +
        formatted_examples
    )

    # gather the prompt components together
    prompts = (
        system.format(max_attempts=max_retries, judgments=labels, dbp_info=dbp_info),
        user,
        retry.format(responses=responses, dbp=dbp)
    )

    return prompts
