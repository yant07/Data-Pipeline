def bar_plot_MulVar(df,x_name,y_name,hue_name):
    sns.catplot(x=x_name, y=y_name, hue=hue_name,kind="bar",data=df)
def bar_plot_UniVar(df,x_name):
    sns.catplot(x=x_name, kind="count", data=df)

def hist_plot(df,IndexName):
    ax1=sns.histplot(df[IndexName])
def dist_plot(df,IndexName):
    ax = sns.distplot(df[IndexName])
def box_plot(df,Index_A,Index_B):
    df.iloc[:,Index_A:Index_B].boxplot()
def sca_plot(df,x_name,y_name):
    sns.scatterplot(data=df, x=x_name, y=y_name)

def line_plot(df,x_name,y_name):
    sns.lineplot(data=flights, x=x_name, y=y_name)

def pair_plot(df,hue_name):
    sns.pairplot(df,hue=hue_name)
def dis_plot_MulVar(df,x_name,hue_name,bins_num):
    sns.displot(df, x=x_name, hue=hue_name,bins=bins_num)