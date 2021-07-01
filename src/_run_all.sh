# python _crawl.py
# python a_format_data.py

python b_graph.py gen_qsubs charts_doubles
../qsubs/b_graph/_commands.sh

python segment.py gen_qsubs charts_doubles
../qsubs/segment/_commands.sh

python c_dijkstra.py gen_qsubs charts_doubles
../qsubs/segment/_commands.sh

python d_annotate.py gen_qsubs charts_doubles
../qsubs/d_annotate/_commands.sh
