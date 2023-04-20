import random
import time

from environment import DHP
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

period_len = 7  # 滚动平均区间
learn_days = 90  # 模拟时长
deck_size = 5000  # 新卡片总量
day_cost_limit = 200
compare_target = 2000
target_halflife = 360

recall_cost = 1
forget_cost = 1
new_cost = 1

policy = []
base = 1.05
min_index = - 30
difficulty_offset = 2
difficulty_limit = 18


def sm2(g, n, ef, i):
    g = int(g)
    if g == 1:
        if n == 0:
            n = 1
            next_i = 1
        elif n == 1:
            n = 2
            next_i = 6
        else:
            n += 1
            next_i = i * ef
        ef += 0.1
    else:
        n = 0
        next_i = 1
        ef += -0.5
    if ef < 1.3:
        ef = 1.3
    return n, ef, next_i


def anki(t, g, n, ef, i):
    g = int(g)
    t = int(t)
    if g == 1:
        if n == 0:
            n = 1
            next_i = 1
        else:
            n += 1
            next_i = (i + (t - i) / 2) * ef
        ef += 0.15
    else:
        n = 0
        next_i = 1
        ef += -0.25
    if ef < 1.3:
        ef = 1.3
    return n, ef, next_i


def scheduler_init():
    for d in range(1, difficulty_limit + 1):
        dataset = pd.read_csv(
            f"./policy/ivl-{d}.csv", header=None, index_col=None)
        global policy
        policy.append(dataset.values)


feature_list = ['difficulty', 'halflife', 'p_recall', 'delta_t', 'reps', 'lapses', 'last_date', 'due_date',
                'r_history', 't_history', 'p_history', 'state', 'cost']

dtypes = np.dtype([
    ('difficulty', int),
    ('halflife', float),
    ('p_recall', float),
    ('delta_t', int),
    ('reps', int),
    ('lapses', int),
    ('last_date', int),
    ('due_date', int),
    ('r_history', str),
    ('t_history', str),
    ('p_history', str),
    ('state', object),
    ('cost', int)
])

field_map = {
    'difficulty': 0, 'halflife': 1, 'p_recall': 2, 'delta_t': 3, 'reps': 4, 'lapses': 5, 'last_date': 6,
    'due_date': 7,
    'r_history': 8,
    't_history': 9,
    'p_history': 10,
    'state': 11,
    'cost': 12}


def scheduler(item: pd.DataFrame, method):
    interval = 1
    t_history = item['t_history'].split(',')
    r_history = item['r_history'].split(',')
    repeats = len(r_history)
    halflife = item['halflife']
    difficulty = item['difficulty']
    if method == "墨墨记忆算法":
        if halflife < target_halflife:
            index = int(np.log(halflife) / np.log(base) - min_index)
            interval = policy[difficulty - 1][index - 1][1]
        else:
            interval = target_halflife
    elif method == "Anki":
        n = 0
        ef = 2.5
        i = 0
        for t, r in zip(t_history, r_history):
            n, ef, i = anki(t, r, n, ef, i)
        interval = i
    elif method == "SM2":
        n = 0
        ef = 2.5
        i = 0
        for t, r in zip(t_history, r_history):
            n, ef, i = sm2(r, n, ef, i)
        interval = i
    elif method == "列表复习周期":
        return [1, 2, 4, 7, 15, np.inf][min(repeats - 1, 5)]
    elif method == "列表复习周期-放回":
        return [1, 2, 4, 7, 15, np.inf][min(reps, 5)]
    return max(1, round(interval + 0.01))


if __name__ == "__main__":
    student = DHP()
    scheduler_init()

    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()
    fig4 = go.Figure()
    fig5 = go.Figure()
    fig6 = go.Figure()
    days = list(range(1, learn_days + 1))
    for method in ("墨墨记忆算法", "列表复习周期", "Anki", "SM2", "列表复习周期-放回"):

        random.seed(42)
        df_memory = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)
        df_memory['difficulty'] = df_memory['difficulty'].map(
            lambda x: random.choices(range(1, 11), weights=(5, 9, 10, 14, 15, 13, 11, 10, 8, 5), k=1)[0])
        df_memory['due_date'] = learn_days

        new_item_per_day = np.array([0.0] * learn_days)
        new_item_per_day_average_per_period = np.array([0.0] * learn_days)
        cost_per_day = np.array([0.0] * learn_days)
        cost_per_day_average_per_period = np.array([0.0] * learn_days)
        learned_per_day = np.array([0.0] * learn_days)
        review_per_day = np.array([0.0] * learn_days)
        recall_per_day = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)
        p_recall_per_day = np.array([0.0] * learn_days)
        meet_target_per_day = np.array([0.0] * learn_days)

        meet_target = 0

        for day in tqdm(range(learn_days)):
            reviewed = 0
            learned = 0
            day_cost = 0

            df_memory["delta_t"] = day - df_memory["last_date"]
            df_memory["p_recall"] = np.exp2(- df_memory["delta_t"] /
                                            df_memory["halflife"])
            need_review = df_memory[df_memory['due_date'] <= day].index
            p_recall_per_day[day] = df_memory[df_memory['due_date']
                                              <= day]['p_recall'].mean()
            for idx in need_review:
                if day_cost >= day_cost_limit:
                    break

                reviewed += 1
                df_memory.iat[idx, field_map['last_date']] = day
                ivl = df_memory.iat[idx, field_map['delta_t']]
                df_memory.iat[idx, field_map['t_history']] += f',{ivl}'

                halflife = df_memory.iat[idx, field_map['halflife']]
                difficulty = df_memory.iat[idx, field_map['difficulty']]
                p_recall = df_memory.iat[idx, field_map['p_recall']]
                df_memory.iat[idx, field_map['p_history']
                ] += f',{p_recall:.2f}'
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]
                state = df_memory.iat[idx, field_map['state']]

                if random.random() < p_recall:
                    day_cost += recall_cost

                    df_memory.iat[idx, field_map['r_history']] += ',1'

                    new_state, new_halflife = student.next_state(
                        state, 1, ivl, p_recall)
                    df_memory.iat[idx, field_map['halflife']] = new_halflife
                    df_memory.iat[idx, field_map['state']] = new_state
                    df_memory.iat[idx, field_map['reps']] = reps + 1
                    df_memory.iat[idx, field_map['cost']] += recall_cost

                    if new_halflife >= target_halflife:
                        meet_target += 1
                        # df_memory.iat[idx, field_map['halflife']] = np.inf
                        # df_memory.iat[idx, field_map['due_date']] = np.inf
                        # continue

                    delta_t = scheduler(df_memory.loc[idx], method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t

                else:
                    day_cost += forget_cost

                    df_memory.iat[idx, field_map['r_history']] += ',0'

                    new_state, new_halflife = student.next_state(
                        state, 0, ivl, p_recall)

                    if new_halflife >= target_halflife:
                        meet_target += 1
                        df_memory.iat[idx, field_map['halflife']] = np.inf
                        df_memory.iat[idx, field_map['due_date']] = np.inf
                        continue

                    df_memory.iat[idx, field_map['halflife']] = new_halflife
                    df_memory.iat[idx, field_map['state']] = new_state

                    reps = 0
                    lapses = lapses + 1

                    df_memory.iat[idx, field_map['reps']] = reps
                    df_memory.iat[idx, field_map['lapses']] = lapses
                    df_memory.iat[idx, field_map['cost']] += forget_cost

                    delta_t = scheduler(df_memory.loc[idx], method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t
                    df_memory.iat[idx, field_map['cost']] += recall_cost

            need_learn = df_memory[df_memory['halflife'].isna()].index

            for idx in need_learn:
                if day_cost >= day_cost_limit:
                    break
                learned += 1
                day_cost += new_cost
                df_memory.iat[idx, field_map['last_date']] = day

                difficulty = df_memory.iat[idx, field_map['difficulty']]
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]

                r, t, p, new_state, new_halflife = student.init(difficulty)

                df_memory.iat[idx, field_map['r_history']] = str(r)
                df_memory.iat[idx, field_map['t_history']] = str(t)
                df_memory.iat[idx, field_map['p_history']] = str(p)
                df_memory.iat[idx, field_map['halflife']] = new_halflife
                df_memory.iat[idx, field_map['state']] = new_state

                delta_t = scheduler(df_memory.loc[idx], method)
                df_memory.iat[idx, field_map['due_date']] = day + delta_t
                df_memory.iat[idx, field_map['cost']] = 0

            new_item_per_day[day] = learned
            learned_per_day[day] = learned_per_day[day - 1] + learned
            cost_per_day[day] = day_cost

            if day >= period_len:
                new_item_per_day_average_per_period[day] = np.true_divide(new_item_per_day[day - period_len:day].sum(),
                                                                          period_len)
                cost_per_day_average_per_period[day] = np.true_divide(cost_per_day[day - period_len:day].sum(),
                                                                      period_len)
            else:
                new_item_per_day_average_per_period[day] = np.true_divide(
                    new_item_per_day[:day + 1].sum(), day + 1)
                cost_per_day_average_per_period[day] = np.true_divide(
                    cost_per_day[:day + 1].sum(), day + 1)

            df_memory["p_recall"] = np.exp2(- df_memory["delta_t"] /
                                            df_memory["halflife"])
            record_per_day[day] = df_memory['p_recall'].sum()
            meet_target_per_day[day] = meet_target

        total_learned = int(sum(new_item_per_day))
        total_cost = int(sum(cost_per_day))

        fig1.add_trace(go.Scatter(x=days, y=record_per_day, name=f'{method}', mode='lines'))
        fig2.add_trace(go.Scatter(x=days, y=meet_target_per_day, name=f'{method}', mode='lines'))
        fig3.add_trace(go.Scatter(x=days, y=new_item_per_day_average_per_period, name=f'{method}', mode='lines'))
        fig4.add_trace(go.Scatter(x=days, y=cost_per_day_average_per_period, name=f'{method}', mode='lines'))
        fig5.add_trace(go.Scatter(x=days, y=learned_per_day, name=f'{method}', mode='lines'))
        fig6.add_trace(go.Scatter(x=days, y=p_recall_per_day, name=f'{method}', mode='lines'))

        print('acc learn', total_learned)
        print('meet target', meet_target)
        print('remembered', record_per_day[-1])

        save = df_memory[df_memory['p_recall'] > 0].copy()
        save['halflife'] = round(save['halflife'], 4)
        save['p_recall'] = round(save['p_recall'], 4)

        save.to_csv(f'./simulation/{method}.tsv', index=False, sep='\t')
        print(new_item_per_day)
        print(learned_per_day)
        print(p_recall_per_day)

    fig1.update_layout(xaxis_title='学习天数', yaxis_title='单词记忆量期望值')
    fig2.update_layout(xaxis_title='学习天数', yaxis_title='达到目标半衰期的单词量')
    fig3.update_layout(xaxis_title='学习天数', yaxis_title=f'每天学习的新单词量 ({period_len} 天平均)')
    fig4.update_layout(xaxis_title='学习天数', yaxis_title=f'每天学习耗时 ({period_len} 天平均)')
    fig5.update_layout(xaxis_title='学习天数', yaxis_title='累积学习单词量')
    fig6.update_layout(xaxis_title='学习天数', yaxis_title='每日学习保留率')

    fig1.write_image("./plot/单词记忆量期望值.pdf", width=1600, height=1000)
    time.sleep(3)
    fig1.write_image("./plot/单词记忆量期望值.pdf", width=1600, height=1000)
    fig2.write_image("./plot/达到目标半衰期的单词量.pdf", width=1600, height=1000)
    time.sleep(3)
    fig2.write_image("./plot/达到目标半衰期的单词量.pdf", width=1600, height=1000)
    fig3.write_image("./plot/每日新单词量.pdf", width=1600, height=1000)
    time.sleep(3)
    fig3.write_image("./plot/每日新单词量.pdf", width=1600, height=1000)
    fig4.write_image("./plot/每日学习耗时.pdf", width=1600, height=1000)
    time.sleep(3)
    fig4.write_image("./plot/每日学习耗时.pdf", width=1600, height=1000)
    fig5.write_image("./plot/累积学习单词量.pdf", width=1600, height=1000)
    time.sleep(3)
    fig5.write_image("./plot/累积学习单词量.pdf", width=1600, height=1000)
    fig6.write_image("./plot/每日学习保留率.pdf", width=1600, height=1000)
    time.sleep(3)
    fig6.write_image("./plot/每日学习保留率.pdf", width=1600, height=1000)
