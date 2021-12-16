import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #kaggle dataset
    pastSeasons = pd.read_csv("all_seasons.csv")
    #datasets from basketball-reference.com
    currSeason = pd.read_csv("perGameStatsRaw.csv")
    currSeasonAdv = pd.read_csv("AdvancedStatsRaw.csv")
    teamWins = pd.read_csv("historicalWins.csv")
    teamWins = teamWins.set_index('Season')

    #drop columns that only show up in the kaggle dataset
    pastSeasonsClean = pastSeasons.drop(columns=['player_weight','player_height','college','country','draft_year','draft_round','draft_number','Unnamed: 0'])

    #fix team abbreviations
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'PHX', 'PHO', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'BKN', 'BRK', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'CHH', 'CHO', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'VAN', 'MEM', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'NJN', 'BRK', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'CHA', 'CHO', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'NOK', 'NOP', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'NOH', 'NOP', inplace=True)
    pastSeasonsClean['team_abbreviation'].mask(pastSeasonsClean['team_abbreviation'] == 'SEA', 'OKC', inplace=True)

    #rename columns to match
    currSeasonClean = currSeason.rename(columns={"Player": "player_name", "Tm": "team_abbreviation", "Age" : "age", "G" : "gp", "TRB":"reb","AST":"ast" , "PTS" : "pts"})



    #make games played a percentage of games played that season
    #get number of gamess played each season
    grouped = pastSeasonsClean.groupby("season")
    n = grouped["gp"].max().to_frame()
    n.reset_index(inplace=True)
    n = n.rename(columns = {'index':'season'})

    #inner join
    pastSeasonsClean = pastSeasonsClean.join(n.set_index('season'), rsuffix='_total',on='season')
    #create new column 
    pastSeasonsClean['gp_pct'] = pastSeasonsClean['gp'] / pastSeasonsClean['gp_total']

    pastSeasonsClean['teamWins'] = 0

    #get team wins for each player
    for i in range(pastSeasonsClean.shape[0]):
        team = pastSeasonsClean.at[i,'team_abbreviation']
        season = pastSeasonsClean.at[i,'season']
        if(team in teamWins.columns):
            wins = teamWins.at[season,team]
            pastSeasonsClean.at[i,'teamWins'] = wins
    
    #win percentage
    pastSeasonsClean['win_pct'] = pastSeasonsClean['teamWins'] / pastSeasonsClean['gp_total']

    #drop unused columns
    pastSeasonsClean = pastSeasonsClean.drop(columns = ['gp','gp_total','teamWins'])



    #some teams have played different amounts of games at this point
    grouped2 = currSeasonClean.groupby("team_abbreviation")
    m = grouped2["gp"].max().to_frame()
    m.reset_index(inplace=True)
    m = m.rename(columns = {'index':'team_abbreviation'})


    currSeasonClean = currSeasonClean.join(m.set_index('team_abbreviation'), rsuffix='_total',on='team_abbreviation')
    #create new column 
    currSeasonClean['gp_pct'] = currSeasonClean['gp'] / currSeasonClean['gp_total']

    currSeasonClean['teamWins'] = 0

    #get team wins for each player
    for i in range(currSeasonClean.shape[0]):
        team = currSeasonClean.at[i,'team_abbreviation']
        if(team in teamWins.columns):
            wins = teamWins.at['2021-22',team]
            currSeasonClean.at[i,'teamWins'] = wins

    #win percentage
    currSeasonClean['win_pct'] = currSeasonClean['teamWins'] / currSeasonClean['gp_total']


    #drop unused columns
    currSeasonClean = currSeasonClean.drop(columns = ['gp','gp_total'])

    #combine stats from advanced and standard basketball reference datasets
    currSeasonClean['ts_pct'] = currSeasonAdv['TS%']
    currSeasonClean['net_rating'] = currSeasonAdv['OBPM'] + currSeasonAdv['DBPM']
    currSeasonClean['usg_pct'] = currSeasonAdv['USG%']
    currSeasonClean['oreb_pct'] = currSeasonAdv['ORB%']
    currSeasonClean['dreb_pct'] = currSeasonAdv['DRB%']
    currSeasonClean['ast_pct'] = currSeasonAdv['AST%']

    #scale these to match kaggle
    currSeasonClean['oreb_pct'] = currSeasonClean['oreb_pct']/100
    currSeasonClean['dreb_pct'] = currSeasonClean['dreb_pct']/100
    currSeasonClean['usg_pct'] = currSeasonClean['usg_pct']/100
    currSeasonClean['ast_pct'] = currSeasonClean['ast_pct']/100

    #reorder dataset for readability (and remove columns not in kaggle dataset)
    currSeasonClean = currSeasonClean[['player_name','team_abbreviation','age','pts','reb','ast','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']]
    pastSeasonsClean = pastSeasonsClean[['player_name','team_abbreviation','season','age','pts','reb','ast','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']]

    #assign MVP values

    pastSeasonsClean['mvp'] = 0
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Karl Malone') & (pastSeasonsClean['season'] == '1996-97'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Michael Jordan') & (pastSeasonsClean['season'] == '1997-98'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Karl Malone') & (pastSeasonsClean['season'] == '1998-99'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Shaquille O\'Neal') & (pastSeasonsClean['season'] == '1999-00'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Allen Iverson') & (pastSeasonsClean['season'] == '2000-01'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Tim Duncan') & (pastSeasonsClean['season'] == '2001-02'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Tim Duncan') & (pastSeasonsClean['season'] == '2002-03'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Kevin Garnett') & (pastSeasonsClean['season'] == '2003-04'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Steve Nash') & (pastSeasonsClean['season'] == '2004-05'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Steve Nash') & (pastSeasonsClean['season'] == '2005-06'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Dirk Nowitzki') & (pastSeasonsClean['season'] == '2006-07'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Kobe Bryant') & (pastSeasonsClean['season'] == '2007-08'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'LeBron James') & (pastSeasonsClean['season'] == '2008-09'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'LeBron James') & (pastSeasonsClean['season'] == '2009-10'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Derrick Rose') & (pastSeasonsClean['season'] == '2010-11'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'LeBron James') & (pastSeasonsClean['season'] == '2011-12'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'LeBron James') & (pastSeasonsClean['season'] == '2012-13'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Kevin Durant') & (pastSeasonsClean['season'] == '2013-14'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Stephen Curry') & (pastSeasonsClean['season'] == '2014-15'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Stephen Curry') & (pastSeasonsClean['season'] == '2015-16'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Russell Westbrook') & (pastSeasonsClean['season'] == '2016-17'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'James Harden') & (pastSeasonsClean['season'] == '2017-18'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Giannis Antetokounmpo') & (pastSeasonsClean['season'] == '2018-19'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Giannis Antetokounmpo') & (pastSeasonsClean['season'] == '2019-20'), 1, inplace=True)
    pastSeasonsClean['mvp'].mask((pastSeasonsClean['player_name'] == 'Nikola Jokic') & (pastSeasonsClean['season'] == '2020-21'), 1, inplace=True)


    pastMVPs = pastSeasonsClean['mvp']

    #drop season column now that MVP has been assigned
    pastSeasonsClean = pastSeasonsClean.drop(columns = ['mvp'])


    #show correlation matrix
    plt.figure(figsize = (15,8))

    corr = pastSeasonsClean[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast','ast_pct','gp_pct','win_pct']].corr(method='pearson')

    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True,
        annot_kws={"fontsize":4}
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    ax.set_title('Heatmap showing Pearson Correlation between features')

    plt.show()

    #features to drop bc of high correlation:
    #ast and ast_pct (makes sense)
    currSeasonClean = currSeasonClean.drop(columns= 'ast')
    pastSeasonsClean = pastSeasonsClean.drop(columns = 'ast')


    #export to csv
    currSeasonClean.to_csv("currSeasonX.csv", index=False)
    pastSeasonsClean.to_csv("pastSeasonsX.csv", index=False)
    pastMVPs.to_csv("pastSeasonsY.csv",index = False)

if __name__ == "__main__":
    main()