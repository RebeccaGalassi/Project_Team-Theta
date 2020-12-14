'''Import all the necessary libraries'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier


def prediction(num):

    loc = 'C:/Users/Utente/Desktop/Data/'
    raw_data_1 = pd.read_csv(loc + '2000-01.csv')
    raw_data_2 = pd.read_csv(loc + '2001-02.csv')
    raw_data_3 = pd.read_csv(loc + '2002-03.csv')
    raw_data_4 = pd.read_csv(loc + '2003-04.csv')
    raw_data_5 = pd.read_csv(loc + '2004-05.csv')
    raw_data_6 = pd.read_csv(loc + '2005-06.csv')
    raw_data_7 = pd.read_csv(loc + '2006-07.csv')
    raw_data_8 = pd.read_csv(loc + '2007-08.csv')
    raw_data_9 = pd.read_csv(loc + '2008-09.csv')
    raw_data_10 = pd.read_csv(loc + '2009-10.csv')
    raw_data_11 = pd.read_csv(loc + '2010-11.csv')
    raw_data_12 = pd.read_csv(loc + '2011-12.csv')
    raw_data_13 = pd.read_csv(loc + '2012-13.csv')
    raw_data_14 = pd.read_csv(loc + '2013-14.csv')
    raw_data_15 = pd.read_csv(loc + '2014-15.csv')
    raw_data_16 = pd.read_csv(loc + '2015-16.csv')

    def parse_date(date):
        if date == '':
            return None
        else:
            return dt.strptime(date, '%d/%m/%y').date()

    def parse_date_other(date):
        if date == '':
            return None
        else:
            return dt.strptime(date, '%d/%m/%Y').date()

    raw_data_1.Date = raw_data_1.Date.apply(parse_date)
    raw_data_2.Date = raw_data_2.Date.apply(parse_date)

    '''The date format for this dataset is different'''

    raw_data_3.Date = raw_data_3.Date.apply(parse_date_other)
    raw_data_4.Date = raw_data_4.Date.apply(parse_date)
    raw_data_5.Date = raw_data_5.Date.apply(parse_date)
    raw_data_6.Date = raw_data_6.Date.apply(parse_date)
    raw_data_7.Date = raw_data_7.Date.apply(parse_date)
    raw_data_8.Date = raw_data_8.Date.apply(parse_date)
    raw_data_9.Date = raw_data_9.Date.apply(parse_date)
    raw_data_10.Date = raw_data_10.Date.apply(parse_date)
    raw_data_11.Date = raw_data_11.Date.apply(parse_date)
    raw_data_12.Date = raw_data_12.Date.apply(parse_date)
    raw_data_13.Date = raw_data_13.Date.apply(parse_date)
    raw_data_14.Date = raw_data_14.Date.apply(parse_date)
    raw_data_15.Date = raw_data_15.Date.apply(parse_date)
    raw_data_16.Date = raw_data_16.Date.apply(parse_date)

    columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

    playing_statistics_1 = raw_data_1[columns_req]
    playing_statistics_2 = raw_data_2[columns_req]
    playing_statistics_3 = raw_data_3[columns_req]
    playing_statistics_4 = raw_data_4[columns_req]
    playing_statistics_5 = raw_data_5[columns_req]
    playing_statistics_6 = raw_data_6[columns_req]
    playing_statistics_7 = raw_data_7[columns_req]
    playing_statistics_8 = raw_data_8[columns_req]
    playing_statistics_9 = raw_data_9[columns_req]
    playing_statistics_10 = raw_data_10[columns_req]
    playing_statistics_11 = raw_data_11[columns_req]
    playing_statistics_12 = raw_data_12[columns_req]
    playing_statistics_13 = raw_data_13[columns_req]
    playing_statistics_14 = raw_data_14[columns_req]
    playing_statistics_15 = raw_data_15[columns_req]
    playing_statistics_16 = raw_data_16[columns_req]

    '''Return goals scored and conceded
        at the end of matchweek arranged
        by teams and matchweek
        Argument passed: playing_stat'''

    def get_goals_scored(playing_stat):
        teams = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            teams[i] = []

        for i in range(len(playing_stat)):
            HTGS = playing_stat.iloc[i]['FTHG']
            ATGS = playing_stat.iloc[i]['FTAG']
            teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
            teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

        GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
        '''Create a dataframe for goals scored where
           rows are teams and cols are matchweek'''

        GoalsScored[0] = 0
        for i in range(2, 39):
            GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        return GoalsScored

    def get_goals_conceded(playing_stat):
        teams = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        for i in range(len(playing_stat)):
            ATGC = playing_stat.iloc[i]['FTHG']
            HTGC = playing_stat.iloc[i]['FTAG']
            teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
            teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
        GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
        GoalsConceded[0] = 0
        for i in range(2, 39):
            GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
        return GoalsConceded

    def get_gss(playing_stat):
        GC = get_goals_conceded(playing_stat)
        GS = get_goals_scored(playing_stat)

        j = 0
        HTGS = []
        ATGS = []
        HTGC = []
        ATGC = []

        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTGS.append(GS.loc[ht][j])
            ATGS.append(GS.loc[at][j])
            HTGC.append(GC.loc[ht][j])
            ATGC.append(GC.loc[at][j])

            if ((i + 1) % 10) == 0:
                j = j + 1

        playing_stat['HTGS'] = HTGS
        playing_stat['ATGS'] = ATGS
        playing_stat['HTGC'] = HTGC
        playing_stat['ATGC'] = ATGC

        return playing_stat

    playing_statistics_1 = get_gss(playing_statistics_1)
    playing_statistics_2 = get_gss(playing_statistics_2)
    playing_statistics_3 = get_gss(playing_statistics_3)
    playing_statistics_4 = get_gss(playing_statistics_4)
    playing_statistics_5 = get_gss(playing_statistics_5)
    playing_statistics_6 = get_gss(playing_statistics_6)
    playing_statistics_7 = get_gss(playing_statistics_7)
    playing_statistics_8 = get_gss(playing_statistics_8)
    playing_statistics_9 = get_gss(playing_statistics_9)
    playing_statistics_10 = get_gss(playing_statistics_10)
    playing_statistics_11 = get_gss(playing_statistics_11)
    playing_statistics_12 = get_gss(playing_statistics_12)
    playing_statistics_13 = get_gss(playing_statistics_13)
    playing_statistics_14 = get_gss(playing_statistics_14)
    playing_statistics_15 = get_gss(playing_statistics_15)
    playing_statistics_16 = get_gss(playing_statistics_16)

    def get_points(result):
        '''Getting respective points'''

        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0

    def get_cuml_points(matchres):
        matchres_points = matchres.applymap(get_points)
        for i in range(2, 39):
            matchres_points[i] = matchres_points[i] + matchres_points[i-1]

        matchres_points.insert(column=0, loc=0, value=[0*i for i in range(20)])
        return matchres_points

    def get_matchres(playing_stat):
        teams = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            teams[i] = []

        for i in range(len(playing_stat)):
            if playing_stat.iloc[i].FTR == 'H':
                teams[playing_stat.iloc[i].HomeTeam].append('W')
                teams[playing_stat.iloc[i].AwayTeam].append('L')
            elif playing_stat.iloc[i].FTR == 'A':
                teams[playing_stat.iloc[i].AwayTeam].append('W')
                teams[playing_stat.iloc[i].HomeTeam].append('L')
            else:
                teams[playing_stat.iloc[i].AwayTeam].append('D')
                teams[playing_stat.iloc[i].HomeTeam].append('D')

        return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

    def get_agg_points(playing_stat):
        matchres = get_matchres(playing_stat)
        cum_pts = get_cuml_points(matchres)
        HTP = []
        ATP = []
        j = 0
        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTP.append(cum_pts.loc[ht][j])
            ATP.append(cum_pts.loc[at][j])

            if ((i + 1) % 10) == 0:
                j = j + 1

        playing_stat['HTP'] = HTP
        playing_stat['ATP'] = ATP
        return playing_stat

    playing_statistics_1 = get_agg_points(playing_statistics_1)
    playing_statistics_2 = get_agg_points(playing_statistics_2)
    playing_statistics_3 = get_agg_points(playing_statistics_3)
    playing_statistics_4 = get_agg_points(playing_statistics_4)
    playing_statistics_5 = get_agg_points(playing_statistics_5)
    playing_statistics_6 = get_agg_points(playing_statistics_6)
    playing_statistics_7 = get_agg_points(playing_statistics_7)
    playing_statistics_8 = get_agg_points(playing_statistics_8)
    playing_statistics_9 = get_agg_points(playing_statistics_9)
    playing_statistics_10 = get_agg_points(playing_statistics_10)
    playing_statistics_11 = get_agg_points(playing_statistics_11)
    playing_statistics_12 = get_agg_points(playing_statistics_12)
    playing_statistics_13 = get_agg_points(playing_statistics_13)
    playing_statistics_14 = get_agg_points(playing_statistics_14)
    playing_statistics_15 = get_agg_points(playing_statistics_15)
    playing_statistics_16 = get_agg_points(playing_statistics_16)

    '''Getting the team form
       Argument passed:
       playing_stat and num'''

    def get_form(playing_stat, num):
        form = get_matchres(playing_stat)
        form_final = form.copy()
        for i in range(num, 39):
            form_final[i] = ''
            j = 0
            while j < num:
                form_final[i] += form[i-j]
                j += 1
        return form_final

    def add_form(playing_stat, num):
        form = get_form(playing_stat, num)
        h = ['M' for i in range(num * 10)]
        a = ['M' for i in range(num * 10)]

        j = num
        for i in range((num*10), 380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            past = form.loc[ht][j]
            h.append(past[num-1])

            past = form.loc[at][j]
            a.append(past[num-1])

            if ((i + 1) % 10) == 0:
                j = j + 1

        playing_stat['HM' + str(num)] = h
        playing_stat['AM' + str(num)] = a

        return playing_stat

    def add_form_df(playing_statistics):
        playing_statistics = add_form(playing_statistics, 1)
        playing_statistics = add_form(playing_statistics, 2)
        playing_statistics = add_form(playing_statistics, 3)
        playing_statistics = add_form(playing_statistics, 4)
        playing_statistics = add_form(playing_statistics, 5)
        return playing_statistics

    playing_statistics_1 = add_form_df(playing_statistics_1)
    playing_statistics_2 = add_form_df(playing_statistics_2)
    playing_statistics_3 = add_form_df(playing_statistics_3)
    playing_statistics_4 = add_form_df(playing_statistics_4)
    playing_statistics_5 = add_form_df(playing_statistics_5)
    playing_statistics_6 = add_form_df(playing_statistics_6)
    playing_statistics_7 = add_form_df(playing_statistics_7)
    playing_statistics_8 = add_form_df(playing_statistics_8)
    playing_statistics_9 = add_form_df(playing_statistics_9)
    playing_statistics_10 = add_form_df(playing_statistics_10)
    playing_statistics_11 = add_form_df(playing_statistics_11)
    playing_statistics_12 = add_form_df(playing_statistics_12)
    playing_statistics_13 = add_form_df(playing_statistics_13)
    playing_statistics_14 = add_form_df(playing_statistics_14)
    playing_statistics_15 = add_form_df(playing_statistics_15)
    playing_statistics_16 = add_form_df(playing_statistics_16)

    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
            'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']

    playing_statistics_1 = playing_statistics_1[cols]
    playing_statistics_2 = playing_statistics_2[cols]
    playing_statistics_3 = playing_statistics_3[cols]
    playing_statistics_4 = playing_statistics_4[cols]
    playing_statistics_5 = playing_statistics_5[cols]
    playing_statistics_6 = playing_statistics_6[cols]
    playing_statistics_7 = playing_statistics_7[cols]
    playing_statistics_8 = playing_statistics_8[cols]
    playing_statistics_9 = playing_statistics_9[cols]
    playing_statistics_10 = playing_statistics_10[cols]
    playing_statistics_11 = playing_statistics_11[cols]
    playing_statistics_12 = playing_statistics_12[cols]
    playing_statistics_13 = playing_statistics_13[cols]
    playing_statistics_14 = playing_statistics_14[cols]
    playing_statistics_15 = playing_statistics_15[cols]
    playing_statistics_16 = playing_statistics_16[cols]

    Standings = pd.read_csv(loc + "EPLStandings.csv")
    Standings.set_index(['Team'], inplace=True)
    Standings = Standings.fillna(18)

    '''Get Last Year's Position
         as an independent variable
        Argument passed:
         playing_stat
         Standings
         year'''

    def get_last(playing_stat, Standings, year):
        HomeTeamLP = []
        AwayTeamLP = []
        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HomeTeamLP.append(Standings.loc[ht][year])
            AwayTeamLP.append(Standings.loc[at][year])
        playing_stat['HomeTeamLP'] = HomeTeamLP
        playing_stat['AwayTeamLP'] = AwayTeamLP
        return playing_stat

    playing_statistics_1 = get_last(playing_statistics_1, Standings, 0)
    playing_statistics_2 = get_last(playing_statistics_2, Standings, 1)
    playing_statistics_3 = get_last(playing_statistics_3, Standings, 2)
    playing_statistics_4 = get_last(playing_statistics_4, Standings, 3)
    playing_statistics_5 = get_last(playing_statistics_5, Standings, 4)
    playing_statistics_6 = get_last(playing_statistics_6, Standings, 5)
    playing_statistics_7 = get_last(playing_statistics_7, Standings, 6)
    playing_statistics_8 = get_last(playing_statistics_8, Standings, 7)
    playing_statistics_9 = get_last(playing_statistics_9, Standings, 8)
    playing_statistics_10 = get_last(playing_statistics_10, Standings, 9)
    playing_statistics_11 = get_last(playing_statistics_11, Standings, 10)
    playing_statistics_12 = get_last(playing_statistics_12, Standings, 11)
    playing_statistics_13 = get_last(playing_statistics_13, Standings, 12)
    playing_statistics_14 = get_last(playing_statistics_14, Standings, 13)
    playing_statistics_15 = get_last(playing_statistics_15, Standings, 14)
    playing_statistics_16 = get_last(playing_statistics_16, Standings, 15)

    def get_mw(playing_stat):
        '''Getting the match week
            Argument passed:
            playing_stat'''

        j = 1
        MatchWeek = []
        for i in range(380):
            MatchWeek.append(j)
            if ((i + 1) % 10) == 0:
                j = j + 1

        playing_stat['MW'] = MatchWeek
        return playing_stat

    playing_statistics_1 = get_mw(playing_statistics_1)
    playing_statistics_2 = get_mw(playing_statistics_2)
    playing_statistics_3 = get_mw(playing_statistics_3)
    playing_statistics_4 = get_mw(playing_statistics_4)
    playing_statistics_5 = get_mw(playing_statistics_5)
    playing_statistics_6 = get_mw(playing_statistics_6)
    playing_statistics_7 = get_mw(playing_statistics_7)
    playing_statistics_8 = get_mw(playing_statistics_8)
    playing_statistics_9 = get_mw(playing_statistics_9)
    playing_statistics_10 = get_mw(playing_statistics_10)
    playing_statistics_11 = get_mw(playing_statistics_11)
    playing_statistics_12 = get_mw(playing_statistics_12)
    playing_statistics_13 = get_mw(playing_statistics_13)
    playing_statistics_14 = get_mw(playing_statistics_14)
    playing_statistics_15 = get_mw(playing_statistics_15)
    playing_statistics_16 = get_mw(playing_statistics_16)

    playing_stat = pd.concat([playing_statistics_1,
                              playing_statistics_2, playing_statistics_3, playing_statistics_4, playing_statistics_5,
                              playing_statistics_6,
                              playing_statistics_7,
                              playing_statistics_8,
                              playing_statistics_9,
                              playing_statistics_10,
                              playing_statistics_11,
                              playing_statistics_12,
                              playing_statistics_13,
                              playing_statistics_14,
                              playing_statistics_15,
                              playing_statistics_16],  ignore_index=True)

    def get_form_points(string):  # Gets the form points.
        sum = 0
        for letter in string:
            sum += get_points(letter)
        return sum

    playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
    playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']

    playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
    playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

    '''Get the string of the form of the 3 or 5
        previous weeks and transform
        the letters in points '''

    def get_3game_ws(string):
        if string[-3:] == 'WWW':
            return 1
        else:
            return 0

    def get_5game_ws(string):
        if string == 'WWWWW':
            return 1
        else:
            return 0

    def get_3game_ls(string):
        if string[-3:] == 'LLL':
            return 1
        else:
            return 0

    def get_5game_ls(string):
        if string == 'LLLLL':
            return 1
        else:
            return 0

    playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
    playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
    playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
    playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

    playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
    playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
    playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
    playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

    playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
    playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

    playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
    playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

    playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']

    cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
    playing_stat.MW = playing_stat.MW.astype(float)

    for col in cols:
        playing_stat[col] = playing_stat[col] / playing_stat.MW

    df = playing_stat.copy()

    odd_data = pd.DataFrame(None)

    odd_data = df[["MW", "HomeTeam", "AwayTeam", "Date"]]
    df = df.drop(["MW", "HomeTeam",  "AwayTeam", "Date",
                  "HM1", "HM2", "HM3", "HM4", "HM5", "AM1",
                  "AM2", "AM3", "AM4", "AM5",
                  "HTFormPtsStr", "ATFormPtsStr"], 1)
    odd_data = odd_data[5700:]

    test_data = df[5700:]

    train_data = df[:5700]

    X_test = test_data.drop('FTR', 1)  # Dropping the "Result" column
    X_train = train_data.drop('FTR', 1)

    y_train = train_data.FTR
    XGB = XGBClassifier(random_state=42)
    XGB.fit(X_train, y_train)

    y_pred_test = XGB.predict(X_test)  # Predict

    ''' Predict the probabilities of each class '''
    y_pred_probability = XGB.predict_proba(X_test)

    test_data['Prediction'] = y_pred_test

    df_prob = pd.DataFrame({'home_prob': [row[2] for row in y_pred_probability],
                            'draw_prob': [row[1] for row in y_pred_probability],
                            'away_prob': [row[0] for row in y_pred_probability]})

    test_data = test_data[['FTR', 'Prediction']]

    dataset = pd.concat([test_data.reset_index(drop=True),
                        odd_data.reset_index(drop=True), df_prob], 1)

    dataset = dataset.drop(["FTR", "Date"], 1)

    '''Producing Final dataframe'''

    dataset = dataset.loc[(dataset['MW'] == num)]
    return dataset