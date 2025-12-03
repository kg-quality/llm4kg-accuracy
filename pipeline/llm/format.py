import re

def check_response(response):
    """
    Check if response is one of the allowed answers

    :param response: the response from the LLM
    :return: True if the response is one of the allowed answers, False otherwise
    """

    labels = ['t', 'f']
    return response.lower() in labels


def convert_response(response):
    """
    Convert the LLM response into the corresponding predefined label

    :param response: the LLM response
    :return: predefined label
    """

    match response.lower():
        case 't':
            return 'true'
        case 'f':
            return 'false'
        case _:
            return 'na'


def format_rdf_item(item):
    """
    Format an RDF item into a "cleaned" string

    :param item: the RDF item to clean
    :return: the cleaned RDF item
    """

    if item.startswith("<") and item.endswith(">"):  # URI case -- strip < > and take last part
        item = item[1:-1]  # remove angle brackets
        item = re.split(r'[\/#]', item)[-1]  # last part after / or #
        item = item.replace('_', ' ')  # underscores to spaces
    else:  # Literal case -- remove datatype annotations like ^^<xsd:date>
        item = re.sub(r'\^\^<.*$', '', item)
        item = item.strip('"')
    return item.strip()


def prepare_fact(fact, dataset):
    """
    Prepare fact depending on the KG dataset

    :param fact: the fact to prepare
    :param dataset: the KG dataset containing the fact
    :return: the prepared fact
    """

    if 'YAGO4.5' in dataset:  # remove YAGO4.5 dataset version -- not required
        dataset = dataset.split('_')[0]

    match dataset:
        case 'DBPEDIA':
            prepared_fact = [str(item).split('/')[-1].replace('_', ' ') for item in fact]
            return {'s': prepared_fact[0], 'p': prepared_fact[1], 'o': prepared_fact[2]}
        case 'FACTBENCH':  # FACTBENCH provides facts in natural language format -- keep them as is
            prepared_fact = fact
            return {'s': prepared_fact[0], 'p': prepared_fact[1], 'o': prepared_fact[2]}
        case 'NELL':
            prepared_fact = [str(item).split(':')[-1].replace('_', ' ') for item in fact]
            return {'s': prepared_fact[0], 'p': prepared_fact[1], 'o': prepared_fact[2]}
        case 'YAGO4.5':  # YAGO4.5 requires ad hoc preparation as it is encoded in RDF
            prepared_fact = [format_rdf_item(item) for item in fact]
            return {'s': prepared_fact[0], 'p': prepared_fact[1], 'o': prepared_fact[2]}
        case _:
            raise ValueError(f'Dataset "{dataset}" not recognized')
