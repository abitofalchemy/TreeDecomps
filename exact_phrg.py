# make the other metrics work
# generate the txt files, then work on the pdf otuput
__version__ = "0.1.0"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
import PHRG
import probabilistic_cfg as pcfg
import net_metrics as metrics
# import dataframe_from_temporal_edgelist as tdf
import pprint as pp
import argparse, traceback

DBG = False

def get_parser ():
  parser = argparse.ArgumentParser(description='exact_phrg: infer a model given a graph (derive a model')
  parser.add_argument('g_fname', metavar='G_FNAME', nargs=1, help='Filename of edgelist graph')
  parser.add_argument('--chunglu',  help='Generate chunglu graphs',action='store_true')
  parser.add_argument('--kron',  help='Generate Kronecker product graphs',action='store_true')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

def Hstar_Graphs_Control (G, graph_name, axs):
  # Derive the prod rules in a naive way, where
  prod_rules = PHRG.probabilistic_hrg_learning(G)
  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  num_nodes = G.number_of_nodes()

  print "Starting max size", 'n=', num_nodes
  g.set_max_size(num_nodes)

  print "Done with max size"

  Hstars = []

  num_samples = 20
  print '*' * 40
  for i in range(0, num_samples):
    rule_list = g.sample(num_nodes)
    hstar = PHRG.grow(rule_list, g)[0]
    Hstars.append(hstar)

  # if 0:
  #   g = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=['ts'])
  #   draw_degree_whole_graph(g,axs)
  #   draw_degree(Hstars, axs=axs, col='r')
  #   #axs.set_title('Rules derived by ignoring time')
  #   axs.set_ylabel('Frequency')
  #   axs.set_xlabel('degree')

  if 1:
    # metricx = [ 'degree','hops', 'clust', 'assort', 'kcore','eigen','gcd']
    metricx = ['degree', 'gcd']
    # g = nx.from_pandas_dataframe(df, 'src', 'trg',edge_attr=['ts'])
    # graph_name = os.path.basename(f_path).rstrip('.tel')
    if DBG: print ">", graph_name
    metrics.network_properties([G], metricx, Hstars, name=graph_name, out_tsv=True)


def pandas_dataframes_from_edgelists (el_files):
  if (el_files is None):  return
  list_of_dataframes = []
  for f in el_files:
    print '~' * 80
    print f
    temporal_graph = False
    with open(f, 'r') as ifile:
      line = ifile.readline()
      while (not temporal_graph):
        if ("%" in line):
          line = ifile.readline()
        elif len(line.split()) > 3:
          temporal_graph = True
    if (temporal_graph):
      dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1, 3], autostrip=True)
      df = pd.DataFrame(dat, columns=['src', 'trg', 'ts'])
    else:
      dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1], autostrip=True)
      df = pd.DataFrame(dat, columns=['src', 'trg'])
    df = df.drop_duplicates()
    list_of_dataframes.append(df)

  return list_of_dataframes

def grow_exact_size_hrg_graphs_from_prod_rules(prod_rules, gname, n, runs=1):
  """
  Args:
    rules: production rules (model)
    gname: graph name
    n:     target graph order (number of nodes)
    runs:  how many graphs to generate
  Returns: list of synthetic graphs

  """
  if n <=0: sys.exit(1)


  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  # # mask = (pddf['ts'] >= pddf['ts'].min()+ span*kSlice) & (pddf['ts'] < pddf['ts'].min()+ span*(kSlice +1))
  # mask = (pddf['ts'] >= pddf['ts'].min()) & (pddf['ts'] < pddf['ts'].min() + span * (kSlice + 1))
  # ldf = pddf.loc[mask]
  #
  # G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])
  #
  num_nodes = n
  if DBG: print "Starting max size"
  g.set_max_size(num_nodes)
  if DBG: print "Done with max size"
  #
  # num_samples = 20
  if DBG: print '*' * 40
  hstars_lst = []
  for i in range(0, runs):
    rule_list = g.sample(num_nodes)
    hstar = PHRG.grow(rule_list, g)[0]
    hstars_lst.append(hstar)
  return hstars_lst

def pwrlaw_plot (xdata, ydata, yerr):
    from scipy import linspace, randn, log10, optimize, sqrt

    powerlaw = lambda x, amp, index: amp * (x**index)

    logx = log10(xdata)
    logy = log10(ydata)
    logyerr = yerr / ydata

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = sqrt( covar[0][0] )
    ampErr = sqrt( covar[1][1] ) * amp

    print index

    # ########
    # plotting
    # ########
    # ax.plot(ydata)
    # ax.plot(pl_sequence)

    fig, axs = plt.subplots(2,1)

    axs[0].plot(xdata, powerlaw(xdata, amp, index))     # Fit
    axs[0].errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    (yh1,yh2) = (axs[0].get_ylim()[1]*.9, axs[0].get_ylim()[1]*.8)
    xh = axs[0].get_xlim()[0]*1.1
    print axs[0].get_ylim()
    print (yh1,yh2)

    axs[0].text(xh, yh1, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    axs[0].text(xh, yh2, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    axs[0].set_title('Best Fit Power Law')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    # xlim(1, 11)
    #
    # subplot(2, 1, 2)
    axs[1].loglog(xdata, powerlaw(xdata, amp, index))
    axs[1].errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    axs[1].set_xlabel('X (log scale)')
    axs[1].set_ylabel('Y (log scale)')

    import datetime
    figfname = datetime.datetime.now().strftime("%d%b%y")+"_pl"
    plt.savefig(figfname, bbox_inches='tight')
    return figfname

def deg_vcnt_to_disk(orig_graph, synthetic_graphs):
    df = pd.DataFrame(orig_graph.degree().items())
    gb = df.groupby([1]).count()
    # gb.to_csv("Results/deg_orig_"+orig_graph.name+".tsv", sep='\t', header=True)
    gb.index.rename('k',inplace=True)
    gb.columns=['vcnt']
    gb.to_csv("Results/deg_orig_"+orig_graph.name+".tsv", sep='\t', header=True)
    # ## - group of synth graphs -
    deg_df = pd.DataFrame()
    for g in synthetic_graphs:
        d  = g.degree()
        df = pd.DataFrame.from_dict(d.items())
        gb = df.groupby(by=[1]).count()
        # Degree vs cnt
        deg_df = pd.concat([deg_df, gb], axis=1)  # Appends to bottom new DFs
    # print gb
    deg_df['mean'] = deg_df.mean(axis=1)
    deg_df.index.rename('k',inplace=True)
    deg_df['mean'].to_csv("Results/deg_xphrg_"+orig_graph.name+".tsv", sep='\t', header=True)

def plot_g_hstars(orig_graph, synthetic_graphs):
    df = pd.DataFrame(orig_graph.degree().items())
    gb = df.groupby([1]).count()
    # gb.to_csv("Results/deg_orig_"+orig_graph.name+".tsv", sep='\t', header=True)
    gb.index.rename('k',inplace=True)
    gb.columns=['vcnt']

    # k_cnt = [(x.tolist(),y.values[0]) for x,y in gb.iterrows()]
    xdata = np.array([x.tolist()  for x,y in gb.iterrows()])
    ydata = np.array([y.values[0] for x,y in gb.iterrows()])
    yerr = ydata *0.000001

    fig, ax = plt.subplots()
    ax.plot(gb.index.values, gb['vcnt'].values,'-o', markersize=8, markerfacecolor='w', markeredgecolor=[0,0,1], alpha=0.5, label="orig")

    ofname = pwrlaw_plot(xdata, ydata,yerr)
    if os.path.exists(ofname): print '... Plot save to:',ofname


    deg_df = pd.DataFrame()
    for g in synthetic_graphs:
        d  = g.degree()
        df = pd.DataFrame.from_dict(d.items())
        gb = df.groupby(by=[1]).count()
        # Degree vs cnt
        deg_df = pd.concat([deg_df, gb], axis=1)  # Appends to bottom new DFs
    # print gb
    deg_df['mean'] = deg_df.mean(axis=1)
    deg_df.index.rename('k',inplace=True)
    # ax.plot(y=deg_df.mean(axis=1))
    # ax.plot(y=deg_df.median(axis=1))
    # ax.plot()
    # orig
    deg_df.mean(axis=1).plot(ax=ax,label='mean',color='r')
    deg_df.median(axis=1).plot(ax=ax,label='median',color='g')
    ax.fill_between(deg_df.index, deg_df.mean(axis=1) - deg_df.sem(axis=1),
                    deg_df.mean(axis=1) + deg_df.sem(axis=1), alpha=0.2, label="se")
    # ax.plot(k_cnt)
    # deg_df.plot(ax=ax)
    # for x,y in k_cnt:
    #     if DBG: print "{}\t{}".format(x,y)
    #
    #
    # for g in synths:
    #     df = pd.DataFrame(g.degree().items())
    #     gb = df.groupby([1]).count()
    #     # gb.plot(ax=ax)
    #     for x,y in k_cnt:
    #         if DBG: print "{}\t{}".format(x,y)
    #
    # # Curve-fit
    #
    plt.savefig('tmpfig', bbox_inches='tight')


def get_hrg_production_rules(edgelist_data_frame, graph_name):
  from growing import derive_prules_from

  df = edgelist_data_frame
  G = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])  # whole graph
  G.name = graph_name
  # pos = nx.spring_layout(G)
  prules = derive_prules_from([G])

  # Synthetic Graphs
  hStars = grow_exact_size_hrg_graphs_from_prod_rules(prules[0], graph_name, G.number_of_nodes(),10)
  print '... hStart graphs:',len(hStars)
  # plot_g_hstars(G,hStars)
  deg_vcnt_to_disk(G, hStars)

  if 1:
      metricx = ['degree']# ,'hops', 'clust', 'assort', 'kcore','eigen','gcd']
      metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)



# if __name__ == '__main__':
#   parser = get_parser()
#   args = vars(parser.parse_args())
#
#   in_file = args['g_fname'][0]
#   datframes = tdf.Pandas_DataFrame_From_Edgelist([in_file])
#   df = datframes[0]
#   # g_name = os.path.basename(in_file).lstrip('out.')
#   g_name = os.path.basename(in_file).split('.')[1]
#
#   print '...', g_name
#
#   if args['chunglu']:
#       print 'Generate chunglu graphs given an edgelist'
#       sys.exit(0)
#   elif args['kron']:
#       print 'Generate chunglu graphs given an edgelist'
#       sys.exit(0)
#
#   try:
#     get_hrg_production_rules(df,g_name)
#   except  Exception, e:
#     print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
#     print str(e)
#     traceback.print_exc()
#     os._exit(1)
#   sys.exit(0)
