from collections import Counter


def get_players_info(fname='input.csv'):
    players = {}
    answers_cnt = {}
    n = -1
    with open(fname, encoding='UTF8') as f:
        for idx, line in enumerate(f):
            words = line.split(',', n)
            words = list(map(lambda x: x.strip().strip('"'), words))
            if idx == 0:
                columns = line.split(',')
                n = len(columns)
                questions = words[5:-1]
                continue
            name = words[1]
            email = words[4]
            answers = words[5:-1]

            players[email] = {
                'name': name,
                'answers': {},
                'idx': idx,
                'score': 0,
                'vectors': []
            }

            for question, answer in zip(questions, answers):
                players[email]['answers'][question] = answer

                if question not in answers_cnt:
                    answers_cnt[question] = []
                answers_cnt[question].append(answer)

        for question, cnt in answers_cnt.items():
            minority_answer = Counter(cnt).most_common()[-1][0]
            print(minority_answer)
            for email, player in players.items():
                if player['answers'][question] == minority_answer:
                    player['score'] += 1
                    player['vectors'].append(1)
                else:
                    player['vectors'].append(-1)

    return players


fname = 'input.csv'
players = get_players_info(fname)
for k, v in players.items():
    print(k, v)
