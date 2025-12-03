# examples for fewshot prompting

# shared examples as these are general purpose KGs -- use them for DBPEDIA, FACTBENCH, and YAGO4.5 KGs
general_examples = [
    {
        'fact': {
            's': 'Monica Bellucci',
            'p': {
                'DBPEDIA': 'birthPlace',
                'FACTBENCH': 'birthPlace',
                'YAGO': 'wasBornIn'
            },
            'o': 'Città di Castello, Italy'
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'Curiosity (rover)',
            'p': {
                'DBPEDIA': 'location',
                'FACTBENCH': 'location',
                'YAGO': 'isLocatedIn'
            },
            'o': 'Mars'
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'Isaac Newton',
            'p': {
                'DBPEDIA': 'knownFor',
                'FACTBENCH': 'knownFor',
                'YAGO': 'isKnownFor'
            },
            'o': 'Universal gravitation'
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'La Rambla',
            'p': {
                'DBPEDIA': 'locationCity',
                'FACTBENCH': 'location',
                'YAGO': 'isLocatedIn'
            },
            'o': 'Berlin'
        },
        'response': {
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'J. K. Rowling',
            'p': {
                'DBPEDIA': 'author',
                'FACTBENCH': 'author',
                'YAGO': 'author'
            },
            'o': 'Lord of The Rings'
        },
        'response': {
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'Hyundai Motor Company',
            'p': {
                'DBPEDIA': 'foundationPlace',
                'FACTBENCH': 'foundationPlace',
                'YAGO': 'locationCreated'
            },
            'o': 'Rio de Janeiro'
        },
        'response': {
            'correctness': 'F'
        }
    }
]

# specialized examples about american sports, teams, and players only -- use them for NELL KG (sports-specific)
sport_examples = [
    {
        'fact': {
            's': 'fernando tatis jr',
            'p': 'athleteplaysforteam',
            'o': 'san diego padres'
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'mike trout',
            'p': 'athleteplaysinleague',
            'o': 'mlb',
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'lebron james',
            'p': 'athleteplaysforteam',
            'o': 'los angeles lakers',
        },
        'response': {
            'correctness': 'T'
        }
    },
    {
        'fact': {
            's': 'pete rose',
            'p': 'athleteplaysforteam',
            'o': 'red sox',
        },
        'response': {
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'red sox',
            'p': 'teamplaysincity',
            'o': 'new york',
        },
        'response': {
            'correctness': 'F'
        }
    },
    {
        'fact': {
            's': 'derek jeter',
            'p': 'athleteplaysforteam',
            'o': 'chicago cubs',
        },
        'response': {
            'correctness': 'F'
        }
    },
]
